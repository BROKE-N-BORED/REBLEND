"""Silent update check + ask-first auto-update (UPD). Qt-free,
stdlib-only, so run.sh can use it before the app starts (via
apply_update.py) and the headless suite can test it without Qt.

How it works, end to end:

- On launch, the window asks GitHub's public releases endpoint
  for the latest Reblend release (one plain GET, no token, no
  identifying information) and compares its tag against the
  running version. Release tags are NOT version-ordered
  (v0.2-r3, v0.2-fcolor, v0.2-sh1), so the comparison is
  equality only: a different tag means an update exists.
- Up to date, or no network, or an answer the check cannot
  read: nothing happens — no popup, no noise. The check's
  failure modes all come back as state "unknown" with a plain
  reason, never an exception (the Pixel Forge v0.69 model).
- An update out: the window asks, and only a Yes starts the
  download. The download goes to a staging folder next to the
  app folder (<app>-update), is checked against the release's
  .sha256 sidecar, and is only then committed (pending.json is
  the commit marker — without it, nothing applies). No sidecar
  means no install, ever: a blind download is refused in plain
  words, as is a checksum mismatch.
- The staged update applies at launch, in run.sh, before the
  app starts — never under a running copy. After Yes to the
  restart prompt, the app spawns a small helper that waits for
  it to exit and re-runs run.sh; after No, the staged update
  simply waits, and the next launch applies it. The apply step
  re-verifies the checksum, extracts over the app folder, and
  removes the staging folder; any failure leaves the old
  version in place and the launch carries on.

Test seams: fetch/download take an injectable urlopen, the
download/stage/apply/restart steps are separate functions, and
request_restart takes an injectable spawn — no test touches
the real network or the real install.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import __version__ as CURRENT_VERSION

REPO = "BROKE-N-BORED/REBLEND"
LATEST_RELEASE_URL = ("https://api.github.com/repos/%s/releases/latest"
                      % REPO)
RELEASES_PAGE_URL = "https://github.com/%s/releases" % REPO
CHECK_TIMEOUT_S = 8
DOWNLOAD_TIMEOUT_S = 60
CHUNK_BYTES = 1 << 16
PENDING_FILE = "pending.json"
STAGED_ZIP_NAME = "update.zip"
DOWNLOAD_TMP_SUFFIX = ".download"
# The launcher scripts, kept executable after every apply: the
# release zip stores their modes, but an update must never be
# able to brick the app by landing a run.sh without its exec
# bit (the desktop entry execs it directly), so the apply
# re-asserts the bit on Reblend's own known scripts whatever
# the zip stored. Nothing else is ever chmodded.
EXECUTABLE_SCRIPTS = (
    "run.sh", "install.sh", "install-reblend.sh",
    "uninstall-reblend.sh", "test.sh", "askpass-zenity.sh",
    "Install Reblend.desktop",
)
# The restart helper waits this long for the old copy to exit
# before launching anyway (a zombie child answers kill -0
# forever, so the wait is bounded).
RESTART_WAIT_S = 10
RESTART_WAIT_STEP_S = 0.2


class UpdaterError(Exception):
    """A plain-words updater failure: network trouble, an
    unreadable answer, a refused install. Never a traceback
    for the user — the message is the message."""


class DownloadCancelled(Exception):
    """The user cancelled the download (raised by the progress
    callback through the download loop)."""


def current_tag():
    """The running version as a release tag ("v0.2-upd").
    The version constant in light_engine/__init__.py always
    equals the tag of the release the tree ships as (without
    the leading v) — if it did not, the app would offer to
    install its own release forever."""
    return "v" + CURRENT_VERSION


def app_dir(app_dir_path=None):
    """The app folder: the tree this module lives in, or the
    given path (tests point it at a temp folder)."""
    if app_dir_path is not None:
        return Path(app_dir_path)
    return Path(__file__).resolve().parent.parent


def staging_dir(app_dir_path=None):
    """Where a downloaded update waits: a folder next to the
    app folder, so replacing the app folder never touches it
    and it survives until the next launch. Installed:
    ~/.local/share/reblend-update. A checkout: <tree>-update."""
    base = app_dir(app_dir_path)
    return base.parent / (base.name + "-update")


def can_restart(app_dir_path=None):
    """True when a restart can relaunch the app: run.sh is
    there to exec. A copy run straight from `python main.py`
    has no launcher to return through, so it defers to the
    next launch instead."""
    return (app_dir(app_dir_path) / "run.sh").exists()


def _pick_assets(payload):
    """The (zip, sha256-sidecar) assets of a release payload,
    as (name, download-url) pairs or None each. The release
    zip is the asset ending in -RELEASE.zip (any .zip is the
    fallback); the sidecar ends in .sha256. GitHub's own
    "Source code" zips are not assets, so they never appear
    here."""
    assets = payload.get("assets") or []
    zip_asset = None
    sha_asset = None
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        if name.endswith(".sha256"):
            sha_asset = (name, url)
        elif name.endswith(".zip"):
            if zip_asset is None or name.endswith("-RELEASE.zip"):
                zip_asset = (name, url)
    return zip_asset, sha_asset


def fetch_latest_release(urlopen=None):
    """The latest release from GitHub's public endpoint, as
    {"tag", "zip_url", "zip_name", "sha_url", "sha_name",
    "page_url"}. Raises UpdaterError with a plain reason when
    the network fails, the answer is not JSON, it carries no
    tag, or it has no program file to download. A missing
    checksum sidecar is NOT an error here — the check can
    still report the update; the install step refuses it.
    urlopen is a test seam."""
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(LATEST_RELEASE_URL,
                    timeout=CHECK_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise UpdaterError(
            "GitHub answered the update check with an error (%s)"
            % exc.code)
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise UpdaterError("the update check timed out")
        raise UpdaterError("could not reach GitHub: %s" % reason)
    except TimeoutError:
        raise UpdaterError("the update check timed out")
    except (ValueError, UnicodeDecodeError):
        raise UpdaterError(
            "GitHub did not answer in a form Reblend understands")
    if not isinstance(payload, dict):
        raise UpdaterError(
            "GitHub did not answer in a form Reblend understands")
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        raise UpdaterError(
            "GitHub did not name a version for the latest release")
    zip_asset, sha_asset = _pick_assets(payload)
    if zip_asset is None:
        raise UpdaterError(
            "the latest release has no program file to download")
    return {"tag": tag.strip(),
            "zip_url": zip_asset[1],
            "zip_name": zip_asset[0],
            "sha_url": sha_asset[1] if sha_asset else None,
            "sha_name": sha_asset[0] if sha_asset else None,
            "page_url": payload.get("html_url") or RELEASES_PAGE_URL}


def check_for_update(current_tag_value=None, urlopen=None):
    """One call, three outcomes, never an exception for a
    network or parsing problem:

    {"state": "newer", "latest": "<tag>", "release": {...}} —
    a release with a different tag is out.
    {"state": "current", "latest": "<tag>", "release": {...}} —
    the running tag is the latest.
    {"state": "unknown", "latest": None, "release": None,
     "reason": "<plain words>"} — the check could not run.

    The comparison is tag equality, not version ordering:
    Reblend tags (v0.2-r3, v0.2-fcolor) do not order."""
    current = (current_tag_value if current_tag_value is not None
               else current_tag())
    try:
        release = fetch_latest_release(urlopen=urlopen)
    except UpdaterError as exc:
        return {"state": "unknown", "latest": None, "release": None,
                "reason": str(exc)}
    if release["tag"] != current.strip():
        return {"state": "newer", "latest": release["tag"],
                "release": release}
    return {"state": "current", "latest": release["tag"],
            "release": release}


def parse_sha256_file(text):
    """The hex digest out of a .sha256 sidecar: either a bare
    hex string or the `sha256sum` form "<hex>  <filename>".
    None when the text carries no usable digest."""
    if not isinstance(text, str):
        return None
    first = text.strip().split()
    if not first:
        return None
    digest = first[0].strip().lower()
    if len(digest) != 64:
        return None
    if any(ch not in "0123456789abcdef" for ch in digest):
        return None
    return digest


def sha256_file(path):
    """The SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip(zip_path, expected_hex):
    """True when the file's digest matches the expected one
    (case-insensitive). A missing or unreadable file is False,
    never an exception."""
    if not isinstance(expected_hex, str):
        return False
    try:
        actual = sha256_file(zip_path)
    except OSError:
        return False
    return actual == expected_hex.strip().lower()


def download(url, dest_path, urlopen=None, progress=None):
    """Download a URL to a file, streamed in chunks. progress,
    when given, is called as progress(downloaded_bytes,
    total_bytes_or_None) after each chunk; a progress callback
    that raises (the cancel path) ends the download at once —
    its exception propagates untouched. Raises UpdaterError
    with a plain reason on any network failure. Returns the
    bytes written."""
    opener = urlopen or urllib.request.urlopen
    try:
        response = opener(url, timeout=DOWNLOAD_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        raise UpdaterError(
            "GitHub answered the download with an error (%s)"
            % exc.code)
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise UpdaterError("the download timed out")
        raise UpdaterError("could not reach GitHub: %s" % reason)
    except TimeoutError:
        raise UpdaterError("the download timed out")
    total = None
    headers = getattr(response, "headers", None) or {}
    try:
        length = headers.get("Content-Length")
    except AttributeError:
        length = None
    if length is not None:
        try:
            total = int(length)
        except (TypeError, ValueError):
            total = None
    written = 0
    try:
        with response:
            with open(dest_path, "wb") as handle:
                while True:
                    try:
                        chunk = response.read(CHUNK_BYTES)
                    except (urllib.error.URLError, TimeoutError) as exc:
                        raise UpdaterError(
                            "the download failed: %s" % exc)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
    except UpdaterError:
        raise
    except OSError as exc:
        raise UpdaterError("could not save the download: %s" % exc)
    return written


def download_and_stage(release, app_dir_path=None, urlopen=None,
                       progress=None):
    """Download a release and stage it for the next launch.
    Refuses (UpdaterError, plain words) when the release has
    no checksum sidecar, when the sidecar cannot be read, or
    when the downloaded zip does not match the checksum —
    never a blind install. On success the staging folder
    holds the verified zip plus pending.json, and the folder
    path is returned. A failed attempt removes its own temp
    files and leaves any previously staged update alone."""
    if not release.get("sha_url"):
        raise UpdaterError(
            "This update has no checksum file, so Reblend will "
            "not install it. An update without a checksum cannot "
            "be checked, and Reblend never installs a download "
            "it cannot check.")
    staging = staging_dir(app_dir_path)
    staging.mkdir(parents=True, exist_ok=True)
    zip_tmp = staging / (STAGED_ZIP_NAME + DOWNLOAD_TMP_SUFFIX)
    sha_tmp = staging / (STAGED_ZIP_NAME + ".sha256"
                         + DOWNLOAD_TMP_SUFFIX)
    try:
        download(release["zip_url"], zip_tmp, urlopen=urlopen,
                 progress=progress)
        download(release["sha_url"], sha_tmp, urlopen=urlopen)
        try:
            sidecar = sha_tmp.read_text(encoding="utf-8",
                                        errors="replace")
        except OSError as exc:
            raise UpdaterError(
                "could not read the update's checksum file: %s"
                % exc)
        expected = parse_sha256_file(sidecar)
        if expected is None:
            raise UpdaterError(
                "The update's checksum file did not contain a "
                "usable checksum, so Reblend will not install it.")
        if not verify_zip(zip_tmp, expected):
            raise UpdaterError(
                "The downloaded update did not match its "
                "checksum, so Reblend will not install it. The "
                "download may be damaged — try again later.")
        # Commit: the verified zip, then pending.json last, so
        # a half-finished stage never looks ready to apply.
        zip_final = staging / STAGED_ZIP_NAME
        os.replace(zip_tmp, zip_final)
        pending = {"tag": release["tag"], "sha256": expected,
                   "zip_name": STAGED_ZIP_NAME}
        (staging / PENDING_FILE).write_text(
            json.dumps(pending, indent=2), encoding="utf-8")
        # The digest lives in pending.json now; the sidecar
        # download has served its purpose.
        try:
            sha_tmp.unlink()
        except OSError:
            pass
    except BaseException:
        for tmp in (zip_tmp, sha_tmp):
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return staging


def pending_update(app_dir_path=None):
    """The staged update's pending.json, or None when nothing
    is staged (or the staging is unreadable — an unreadable
    staging is treated as no update, never a crash)."""
    try:
        raw = (staging_dir(app_dir_path) / PENDING_FILE).read_text(
            encoding="utf-8")
        pending = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(pending, dict):
        return None
    if not isinstance(pending.get("tag"), str):
        return None
    if not isinstance(pending.get("sha256"), str):
        return None
    if not isinstance(pending.get("zip_name"), str):
        return None
    return pending


def _safe_extract(zip_path, target_dir):
    """Extract a zip over a folder: every member must land
    inside the folder (a member escaping it is refused), and
    .venv members are skipped — the environment is never part
    of an update. zipfile restores the stored Unix modes, so
    run.sh stays executable."""
    target = Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        # Validate every member BEFORE extracting any, so a
        # refused zip leaves the app folder untouched.
        for member in members:
            name = member.filename
            if not name or name.endswith("/"):
                continue
            first = name.split("/", 1)[0]
            if first in (".venv", ""):
                continue
            dest = (target / name).resolve()
            if dest != target and target not in dest.parents:
                raise UpdaterError(
                    "The update tried to write outside the app "
                    "folder (%s), so it was refused." % name)
        for member in members:
            name = member.filename
            if not name or name.endswith("/"):
                continue
            if name.split("/", 1)[0] in (".venv", ""):
                continue
            archive.extract(member, target)


def _fail_staging(staging, tag, error):
    """A failed staging is moved aside (one previous failure
    kept at most) so the next launch does not retry it
    forever, and the launch carries on with the old version."""
    failed = staging.parent / (staging.name + ".failed")
    try:
        if failed.exists():
            shutil.rmtree(failed)
        os.replace(staging, failed)
    except OSError:
        pass
    return {"applied": False, "tag": tag, "error": error}


def apply_pending_update(app_dir_path=None):
    """Apply a staged update over the app folder. The launcher
    (run.sh, via apply_update.py) calls this before the app
    starts, so no file changes under a running copy. Returns
    {"applied": bool, "tag", "error"} and never raises: any
    failure leaves the old version in place and the launch
    carries on. A developer checkout (a .git folder in the
    app folder) is never overwritten — the installed app has
    no .git, so this only ever guards a working tree."""
    app = app_dir(app_dir_path)
    staging = staging_dir(app_dir_path)
    pending = pending_update(app_dir_path)
    if pending is None:
        return {"applied": False, "tag": None, "error": None}
    tag = pending["tag"]
    if (app / ".git").exists():
        return _fail_staging(
            staging, tag,
            "This copy of Reblend is a developer checkout, so "
            "the update was not applied over it.")
    zip_path = staging / pending["zip_name"]
    try:
        if not verify_zip(zip_path, pending["sha256"]):
            return _fail_staging(
                staging, tag,
                "The staged update did not match its checksum, "
                "so it was not installed.")
        _safe_extract(zip_path, app)
        # The launcher scripts stay executable whatever the
        # zip stored (see EXECUTABLE_SCRIPTS). Best effort
        # per file — a chmod failure must not fail an apply
        # that already landed.
        for name in EXECUTABLE_SCRIPTS:
            script = app / name
            if script.is_file():
                try:
                    os.chmod(script, 0o755)
                except OSError:
                    pass
    except UpdaterError as exc:
        return _fail_staging(staging, tag, str(exc))
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        return _fail_staging(
            staging, tag, "The staged update could not be "
            "installed: %s" % exc)
    try:
        shutil.rmtree(staging)
    except OSError:
        pass
    try:
        with open(app / "update.log", "a", encoding="utf-8") as log:
            log.write("Updated to %s on %s\n" % (
                tag, datetime.datetime.now().isoformat(
                    timespec="seconds")))
    except OSError:
        pass
    return {"applied": True, "tag": tag, "error": None}


def request_restart(app_dir_path=None, pid=None, spawn=None):
    """Restart the app into the staged update: spawn a helper
    that waits for this process to exit, then re-runs run.sh
    (which applies the staged update and launches the new
    version). The helper is detached and fully independent,
    so it survives the app quitting. Returns True when the
    helper started, False when there is no launcher to return
    through or the spawn failed — the staged update still
    applies at the next launch either way. spawn is a test
    seam (called with the argv list)."""
    app = app_dir(app_dir_path)
    run_sh = app / "run.sh"
    if not run_sh.exists():
        return False
    pid = os.getpid() if pid is None else pid
    steps = int(RESTART_WAIT_S / RESTART_WAIT_STEP_S)
    script = (
        "i=0; while kill -0 \"$1\" 2>/dev/null && "
        "[ \"$i\" -lt %d ]; do sleep %s; i=$((i+1)); done; "
        "exec \"$2\"" % (steps, RESTART_WAIT_STEP_S))
    argv = ["/bin/sh", "-c", script, "reblend-restart",
            str(pid), str(run_sh)]
    spawner = spawn or (lambda cmd: subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True))
    try:
        spawner(argv)
    except OSError:
        return False
    return True
