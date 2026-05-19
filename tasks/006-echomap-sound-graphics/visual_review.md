# Visual Review — Goal 006 (echomap sound graphics)

Reviewer-friendly snapshots of the surface-heatmap + listener-viz pipeline.
All four PNGs are regenerated deterministically by
`cargo test --test renderer_screenshots`, which loads the corresponding
STEP fixture, runs a 10k-ray acoustic sim, and software-rasterizes the
surface overlay (and listener pulse for the studio shot).

The renderer used at runtime is egui-backed; these screenshots use a
minimal CPU rasterizer so the artifacts are reproducible in CI without
standing up an offscreen GL context.

## Box room — band variants

| Band      | File                                                                                                                                                                | Notes                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Broadband | [`tests/fixtures/screenshots/box_room_broadband.png`](../../tests/fixtures/screenshots/box_room_broadband.png) | Full energy across all 6 bands (current scalar passthrough).     |
| 125 Hz    | [`tests/fixtures/screenshots/box_room_125hz.png`](../../tests/fixtures/screenshots/box_room_125hz.png)         | Low-frequency band — air absorption tilt minimal.                |
| 4 kHz     | [`tests/fixtures/screenshots/box_room_4khz.png`](../../tests/fixtures/screenshots/box_room_4khz.png)           | High-frequency band — synthetic air-absorption attenuation 0.3x. |

**Today**: band attenuation is a synthetic tilt applied during PNG
generation so the band selector wiring is visible. **After goal 005**
lands `[f32;6]` per-band energy in the acoustic core, this tilt is
replaced by real per-band grid data; the screenshot generator switches
to `sample_band_energy(gp, band)` directly and the synthetic
`band_attenuation()` helper goes away.

## Studio — listener pulse

| Shot                  | File                                                                                                                                                                          | Notes                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Studio listener pulse | [`tests/fixtures/screenshots/studio_listener_pulse.png`](../../tests/fixtures/screenshots/studio_listener_pulse.png) | Studio scene rendered with the broadband surface heatmap; one listener sphere pulses (radius + cold→hot color) by captured normalized SPL with capture-radius shell. |

## What's exercised

- `surface_heatmap::face_energies` — per-face energy via nearest grid sample
- `surface_heatmap::energy_to_log_db` — 60 dB dynamic-range mapping
- `viridis_color` — perceptual colormap on surfaces
- `FrequencyBand` selector — Broadband / 125 Hz / 4 kHz variants
- `listener_viz::capture_listener_energy` + `normalized_spl` — captured SPL on the listener
- `listener_viz::pulse_radius` + `spl_color` — pulse size and hot/cold tint
- Capture-radius shell — drawn as a transparent ring

## Regeneration

```bash
cargo test --test renderer_screenshots
```

Outputs land in `tests/fixtures/screenshots/`. The test asserts each PNG
exists and is non-trivial (> 1 KiB), and that the directory contains ≥ 4
PNGs.
