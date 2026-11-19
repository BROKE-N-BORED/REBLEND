#!/bin/sh
# Reblend v0.2 tests. Run:  bash test.sh
# The suite runs in groups, not one process: a single run grows too big
# under load (a Qt test process grows about 50 MB per test and gets
# killed), so each group runs in its own process and test.sh fails if
# any group fails. Groups: A2/B2 headless, then the Qt groups when
# LIGHT_ENGINE_QT_TESTS=1 is set (needs a display: xvfb-run -a bash test.sh).
# QtGizmoDragTests runs as two method lists (D1/D2) for the same reason.
cd "$(dirname "$0")"
ROOT="$(pwd)"
if [ -x ".venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi
# PY must be absolute: the groups run from inside tests/, where a
# relative .venv path does not exist.
export PYTHONPATH="$ROOT"
cd tests || exit 1
fail=0
run() {
  echo "=== $1 ==="
  name="$1"
  shift
  if ! "$PY" -m unittest "$@"; then
    echo "FAILED: $name"
    fail=1
  fi
}
run "A2 headless" \
  test_light_engine.SceneTests test_light_engine.GeometryTests \
  test_light_engine.IndexedGeometryTests test_light_engine.SmoothShadingTests \
  test_light_engine.AnimationTests test_light_engine.TextureTests \
  test_light_engine.FileIOTests test_light_engine.ShaderSourceTests \
  test_light_engine.PbrAcesTests test_light_engine.ShadowTests \
  test_light_engine.SceneFormatV2Tests test_light_engine.MeshFileTests \
  test_light_engine.MeshObjectTests test_light_engine.VideoExportTests \
  test_light_engine.ParticleSimTests test_light_engine.ParticleRenderTests \
  test_light_engine.InspectorTextTests test_light_engine.QtFreeImportTests \
  test_light_engine.InstallAndCiTests test_light_engine.StyleDataTests \
  test_light_engine.SteppedParticleTests \
  test_light_engine.MaterialPhysicsTests \
  test_light_engine.MaterialLibraryTests \
  test_light_engine.MaterialParamsTests \
  test_light_engine.MaterialLayoutsTests \
  test_light_engine.MaterialWearTests \
  test_light_engine.UpdaterTests
run "B2 headless" \
  test_light_engine.ObjImportTests test_light_engine.GlbImportTests \
  test_light_engine.ObjExportTests test_light_engine.GlbExportTests \
  test_light_engine.GizmoGeometryTests test_light_engine.GizmoPickTests
if [ "$LIGHT_ENGINE_QT_TESTS" = "1" ]; then
  run "A1 Qt" \
    test_light_engine.QtViewportTests test_light_engine.QtMeshTests \
    test_light_engine.QtGlbImportTests test_light_engine.QtGlbExportTests \
    test_light_engine.QtGizmoUndoTests
  run "C1 QtObjImport" test_light_engine.QtObjImportTests
  run "C2 QtObjExport" test_light_engine.QtObjExportTests
  run "C3 QtGizmo" test_light_engine.QtGizmoTests
  run "D1 QtGizmoDrag first 14" \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_move_handles \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_rotate_handles \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_scale_handles \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_misses_empty_space \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_none_without_selection \
    test_light_engine.QtGizmoDragTests.test_pick_gizmo_none_when_suppressed \
    test_light_engine.QtGizmoDragTests.test_hover_highlights_handle \
    test_light_engine.QtGizmoDragTests.test_hover_clears_when_leaving \
    test_light_engine.QtGizmoDragTests.test_drag_move_x \
    test_light_engine.QtGizmoDragTests.test_drag_move_y \
    test_light_engine.QtGizmoDragTests.test_drag_move_z \
    test_light_engine.QtGizmoDragTests.test_drag_rotate_x \
    test_light_engine.QtGizmoDragTests.test_drag_rotate_y \
    test_light_engine.QtGizmoDragTests.test_drag_rotate_z
  run "D2 QtGizmoDrag remaining 13" \
    test_light_engine.QtGizmoDragTests.test_drag_scale_x \
    test_light_engine.QtGizmoDragTests.test_drag_scale_y \
    test_light_engine.QtGizmoDragTests.test_drag_scale_z \
    test_light_engine.QtGizmoDragTests.test_drag_move_x_exact \
    test_light_engine.QtGizmoDragTests.test_drag_scale_x_exact \
    test_light_engine.QtGizmoDragTests.test_snap_move \
    test_light_engine.QtGizmoDragTests.test_snap_rotate \
    test_light_engine.QtGizmoDragTests.test_snap_scale \
    test_light_engine.QtGizmoDragTests.test_inspector_updates_live_during_drag \
    test_light_engine.QtGizmoDragTests.test_handle_wins_over_orbit \
    test_light_engine.QtGizmoDragTests.test_orbit_still_works_on_empty_space \
    test_light_engine.QtGizmoDragTests.test_press_release_on_handle_keeps_selection \
    test_light_engine.QtGizmoDragTests.test_drag_moves_exact_frame_key
  run "C6 QtVideoExport" test_light_engine.QtVideoExportTests
  run "C7 QtEmitter + QtParticleRender" \
    test_light_engine.QtEmitterTests test_light_engine.QtParticleRenderTests
  run "C8 QtInspector" test_light_engine.QtInspectorTests
  run "C9 QtEmitterEditor" test_light_engine.QtEmitterEditorTests
  run "C10 QtWindowFit" test_light_engine.QtWindowFitTests
  run "C11 QtUnsavedWork" test_light_engine.QtUnsavedWorkTests
  run "C12 QtPlainWords" test_light_engine.QtPlainWordsTests
  run "C13 QtActivity" test_light_engine.QtActivityTests
  run "C14 QtVisualIdentity" test_light_engine.QtVisualIdentityTests
  run "C15 QtFrameStats" test_light_engine.QtFrameStatsTests
  run "C16 QtViewportTuner" test_light_engine.QtViewportTunerTests
  run "C17 QtSteppedParticle" test_light_engine.QtSteppedParticleTests
  run "C18 QtMaterialPhysics" test_light_engine.QtMaterialPhysicsTests
  run "C19 QtMaterialPreview" test_light_engine.QtMaterialPreviewTests
  run "C20 QtMaterialLibrary" test_light_engine.QtMaterialLibraryTests
  run "C21 QtMaterialParams" test_light_engine.QtMaterialParamsTests
  run "C22 QtMaterialSimpleAdvanced" test_light_engine.QtMaterialSimpleAdvancedTests
  run "C23 QtMaterialLayout" test_light_engine.QtMaterialLayoutTests
  run "C24 QtMaterialWear" test_light_engine.QtMaterialWearTests
  # C25 QtPieMenu was retired in R2: the wheel replaces the pie
  # menu, and its behaviours live on in C30 where they have a
  # wheel equivalent. The slot stays vacant; later groups are
  # not renumbered.
  run "C26 QtFireGhost" test_light_engine.QtFireGhostRegressionTests
  run "C27 QtFColor" test_light_engine.QtFColorTests
  run "C28 QtDial" test_light_engine.QtDialTests
  run "C29 QtUpdater" test_light_engine.QtUpdaterTests
  run "C30 QtWheel" test_light_engine.QtWheelTests
fi
if [ "$fail" = "0" ]; then
  echo "All groups passed."
fi
exit $fail
