# Manual smoke — human UI walkthrough

The automated gates in [SMOKE.md](SMOKE.md) cover correctness of the
Rust + Python code paths but cannot exercise the egui GUI (no display
in CI). Before tagging a release, a human runs the 10 steps below in
order and confirms each observable result. Tick the boxes inline; the
release manager keeps the signed-off copy in `tasks/010-echomap-quality-gate/`.

## Setup

```bash
cargo build --release --bin echomap
target/release/echomap
```

The window should open within 3s on a current Mac/Linux/Windows
desktop. If it does not, capture the terminal log and abort the smoke.

## Steps

- [ ] **1. New scene loads clean.**
  Click **File → New Scene**. Expected: viewport clears to dark grey
  background, status bar reads `Ready`, outliner shows zero meshes /
  sources / listeners.

- [ ] **2. STEP file loads.**
  Click **File → Open STEP File…**, pick `test_files/box_room.step`.
  Expected: a 5×4×3 m box room appears in the viewport, focused by the
  camera, status bar shows the loaded filename, outliner lists the
  mesh.

- [ ] **3. Sound source placement.**
  Click **Add → Sound Source**. Expected: an emitter glyph appears at
  the origin, outliner gets a new `Source 1` entry, side panel reveals
  per-source frequency + power controls.

- [ ] **4. Listener placement.**
  Click **Add → Listener**. Expected: a listener glyph appears at the
  origin, outliner gets a new `Listener 1` entry with a capture-radius
  slider.

- [ ] **5. Run acoustic simulation.**
  In the simulation config side panel, set `Ray count = 1000`,
  `Max bounces = 20`. Click **Run**. Expected: progress bar in the
  status bar advances smoothly from 0 → 100 %, viewport shows ray paths
  (if **View → Show Ray Paths** is on), final `Sim complete (Nms)`
  message in status bar.

- [ ] **6. Switch frequency bands.**
  In the results pane, toggle between bands 125 Hz → 4 kHz. Expected:
  the surface heatmap and colorbar legend rescale per band; absorbent
  materials show visibly lower energy at 4 kHz than at 125 Hz.

- [ ] **7. Toggle debug ray visualization.**
  Click **View → Show Sensor Rays** and pick a ray path from the debug
  list. Expected: a single ray is highlighted in the viewport with its
  bounce points marked, no flicker, frame rate stays > 30 fps.

- [ ] **8. Save scene.**
  Click **File → Save Scene…**, write to a temp path. Expected: status
  bar reads `Scene saved (Npath)`; the file on disk is non-empty valid
  JSON whose top level includes a `"version"` field (see
  `SNAPSHOT_VERSION` in `src/ui/scene_io.rs`).

- [ ] **9. Reload scene.**
  Restart the app (`Ctrl/Cmd+Q` then re-launch). Click **File → Open
  Scene…**, pick the file from step 8. Expected: the box room, source,
  listener, and ray-count config all come back identical to pre-save.

- [ ] **10. Export results CSV + screenshot capture.**
  After running the sim again, click **File → Export Results CSV…**
  and write to a temp path. Expected: CSV opens in a spreadsheet,
  header matches
  `x,y,z,energy_125hz,energy_250hz,energy_500hz,energy_1khz,energy_2khz,energy_4khz,broadband`
  (see `CSV_HEADER` in `src/io/export.rs`), row count equals the grid
  cell count. Then take a screenshot of the viewport (system-native
  shortcut) and attach it to the release thread.

## Agent Inspector spot-check (post-goal/008)

- Open the boxing mode (`./target/release/echomap_server` then a Python
  agent), or use the boxing demo flow. In the desktop app go to
  **View → Agent Inspector**. Expected: the window opens with the
  capability badge row (`observe`, `step`, `motors`, `sensors`,
  `messaging`, `combat` for a boxing humanoid), and once an agent
  binds the message lane starts ticking with `▶ bind`, `◀ bound`,
  `▶ step`, `◀ stepped` rows. Click a row to expand its JSON payload.

## Sign-off

```
Date:      ____-__-__
Tester:    __________________
Build sha: __________________
All steps green: [ ] yes  [ ] no — see notes below
Notes:     __________________________________________________
```
