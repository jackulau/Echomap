# EchoMap UX Walkthrough

A 10-step tour of the production-feel desktop UI shipped under goal 007.
Each numbered step assumes you have a fresh launch of `cargo run --release`
and `egui` window open at 1280×800.

0. **Prerequisites.** Install Rust 1.79+, clone the repo, and run
   `cargo build --release` once so the first launch is warm. Have a sample
   `.step` file ready (the `tests/data/` folder ships several).

1. **Launch & first impression.** Start `cargo run --release` from the repo root.
   The window opens in dark theme with a top menu bar (File / Add / View / Help),
   a left side panel with collapsible groups, a 3D viewport, and a bottom status
   bar showing "Idle" on the right. *(screenshot: `01-launch.png`)*

2. **New scene reset.** Click **File → New Scene**. The viewport clears and a
   green "New scene" message appears on the left of the status bar.
   *(screenshot: `02-new-scene.png`)*

3. **Open a STEP model.** Click **File → Open STEP File…**, pick a `.step`
   file from disk. The mesh imports, the camera auto-frames the model, and
   the status bar reports "Loaded STEP: <path>". The Scene Objects group in
   the side panel lists the imported objects with visibility checkboxes.
   *(screenshot: `03-step-load.png`)*

4. **Add a sound source.** Use **Add → Sound Source** or expand the
   **Sources** group in the side panel and adjust position, frequency, and
   power. Hovering over any control reveals a tooltip explaining the units
   and acceptable range. *(screenshot: `04-add-source.png`)*

5. **Configure the simulation.** Expand **Simulation Config** in the side
   panel. Drag the `ray_count` slider (logarithmic, 100..100 000), the
   `max_bounces` slider (0..1000), and the `grid_resolution` slider
   (0.05..2.0 m). Each slider has helper text underneath and an inline error
   in red if you push the value out of bounds — the **Run Simulation** button
   greys out until everything validates. *(screenshot: `05-sim-config.png`)*

6. **Run a simulation.** Click **Run Simulation**. The status bar replaces
   "Idle" with a live percentage progress bar and `N/M rays` overlay; the
   **Cancel** button on the right enables for the duration. When complete,
   the **Results** group populates with grid samples, ray-path count, and
   max-energy. *(screenshot: `06-run-sim.png`)*

7. **Inspect the viewport.** Press **F** to focus the camera on the
   selected object, **R** to reset the camera, **Shift+1/2/3** for Front /
   Top / Side preset views, and **Tab** to toggle WASD fly mode. Drag with
   the left mouse to orbit, scroll to zoom, Shift-drag to pan.
   *(screenshot: `07-camera-shortcuts.png`)*

8. **Save the scene.** Click **File → Save Scene…**, choose a destination.
   EchoMap writes a JSON snapshot containing meshes, sources, listeners,
   background medium, and the current Simulation Config. A green status
   message confirms the path. *(screenshot: `08-save.png`)*

9. **Reload the scene.** Close the app, relaunch, then **File → Load
   Scene…** the JSON you just saved. The full scene returns — geometry,
   sliders, and selection state — and the camera frames the model.
   *(screenshot: `09-load.png`)*

10. **Export results & view About.** With a finished simulation in memory,
    click **File → Export Results…**. EchoMap writes a `results.csv` (one
    row per listener with energy + SPL) and a sibling `results.report.md`
    text summary. Finally, **Help → About EchoMap** opens a modal showing
    the version pulled from `CARGO_PKG_VERSION` and a link to this
    walkthrough. *(screenshot: `10-about-export.png`)*
