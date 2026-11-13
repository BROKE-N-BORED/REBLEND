"""Video export: pipe raw RGB frames to ffmpeg (T15). Stdlib only.

The window renders each frame with the existing render path, converts
the grab to packed RGB24 bytes, and writes the bytes to ffmpeg's
stdin. ffmpeg encodes to MP4 (libx264) or WebM (libvpx-vp9). No video
library is vendored — ffmpeg is an external program the user installs
once with the install line below.

This module is Qt-free on purpose: the subprocess, the command line,
and the cancel path are all headless-testable with a fake ffmpeg.
"""
from __future__ import annotations

import os
import shutil
import subprocess

FFMPEG_INSTALL_LINE = "sudo apt install ffmpeg"

# Scale factors offered in the export dialog, as fractions of the
# viewport size. Aspect is preserved because both axes scale together.
RESOLUTION_SCALES = (1.0, 0.5, 0.25)

FORMATS = {
    "mp4": {
        "label": "MP4 (H.264)",
        "extension": ".mp4",
        "codec": "libx264",
        # x264 quality: CRF 20, medium preset, faststart so the file
        # plays before it has fully downloaded.
        "options": ["-crf", "20", "-preset", "medium",
                    "-movflags", "+faststart"],
    },
    "webm": {
        "label": "WebM (VP9)",
        "extension": ".webm",
        "codec": "libvpx-vp9",
        # VP9 quality mode: CRF 30 with no bitrate target.
        "options": ["-crf", "30", "-b:v", "0"],
    },
}


def find_ffmpeg(ffmpeg="ffmpeg"):
    """The ffmpeg executable on PATH, or None when it is not
    installed. `ffmpeg` is a parameter so tests can probe a name
    that is certainly absent."""
    return shutil.which(ffmpeg)


def even(value):
    """Round down to an even number (yuv420p needs even width and
    height). Never below 2."""
    return max(2, int(value) & ~1)


def export_size(viewport_width, viewport_height, scale):
    """The export size for a scale factor of the viewport size,
    rounded to even dimensions."""
    return (even(viewport_width * scale),
            even(viewport_height * scale))


def build_ffmpeg_command(ffmpeg, output_path, width, height, fps, fmt):
    """The ffmpeg argv for a rawvideo pipe: RGB24 frames of
    (width, height) at fps on stdin, encoded to `output_path`.
    `-nostats -loglevel error` keeps stderr small (ffmpeg writes
    per-frame progress to stderr by default, which would fill the
    pipe and stall the encode); `-y` stops ffmpeg asking about
    overwriting, which would hang a piped run."""
    spec = FORMATS[fmt]
    return [ffmpeg,
            "-nostats", "-loglevel", "error",
            "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", spec["codec"], "-pix_fmt", "yuv420p",
            *spec["options"],
            output_path]


class VideoExporter:
    """One ffmpeg encode: frames in on stdin, a video file out.
    start() spawns ffmpeg; write_frame() feeds it one packed RGB24
    frame; finish() closes stdin and waits for the encode;
    cancel() kills ffmpeg and deletes the partial file."""

    def __init__(self, ffmpeg, output_path, width, height, fps, fmt):
        self.ffmpeg = ffmpeg
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.fmt = fmt
        self.frames_written = 0
        self._proc = None
        self._spawn_error = None

    def start(self):
        """Spawn ffmpeg. False when the spawn itself fails (ffmpeg
        missing or not executable) — the caller reports
        spawn_error() instead of crashing."""
        command = build_ffmpeg_command(self.ffmpeg, self.output_path,
                                       self.width, self.height,
                                       self.fps, self.fmt)
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as exc:
            self._spawn_error = str(exc)
            return False
        return True

    def spawn_error(self):
        return self._spawn_error

    def write_frame(self, rgb_bytes):
        """Write one packed RGB24 frame (width*height*3 bytes).
        False when ffmpeg has died or the pipe broke — the caller
        stops the export and calls finish() for the error text."""
        if self._proc is None or self._proc.stdin is None:
            return False
        try:
            self._proc.stdin.write(rgb_bytes)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        self.frames_written += 1
        return True

    def finish(self):
        """Close stdin, wait for ffmpeg, return (ok, stderr text).
        ok is False when ffmpeg exited non-zero; the stderr text
        (short, because of -loglevel error) names the failure."""
        if self._proc is None:
            return False, self._spawn_error or "ffmpeg was not started"
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            return False, "ffmpeg did not finish after stdin closed"
        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
        if self._proc.returncode != 0:
            return False, stderr.decode("utf-8", "replace").strip()
        return True, ""

    def cancel(self):
        """Kill ffmpeg and delete the partial output file, so a
        cancelled export leaves nothing behind. Best effort throughout
        — a missing file or an already-dead process is not an error."""
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            os.unlink(self.output_path)
        except OSError:
            pass
