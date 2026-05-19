//! Result export — CSV (per-grid-point band energies) and text reports.
//!
//! Schema is fixed: the CSV header is exact and consumers parse positionally.
//! Changing it is a versioned data-contract change, not a refactor.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::acoustics::simulation::{SimulationConfig, SimulationResult};

/// Exact CSV header — must match the deliverable schema byte-for-byte.
pub const CSV_HEADER: &str =
    "x,y,z,energy_125hz,energy_250hz,energy_500hz,energy_1khz,energy_2khz,energy_4khz,broadband";

/// Sanitize a float for CSV output — non-finite (NaN / Inf) becomes 0.0
/// so the file is always parseable. Real callers shouldn't be emitting
/// non-finite values, but defensive sanitisation here means a single
/// bad grid cell can't corrupt the entire export.
#[inline]
fn safe_f32(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Errors surfaced by the export pipeline. Distinguished so the UI can show
/// a clear message when the user clicks Export before running a sim.
#[derive(Debug)]
pub enum ExportError {
    NoResults,
    Io(io::Error),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::NoResults => write!(
                f,
                "No simulation results available — run a simulation before exporting"
            ),
            ExportError::Io(e) => write!(f, "I/O error during export: {e}"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<io::Error> for ExportError {
    fn from(e: io::Error) -> Self {
        ExportError::Io(e)
    }
}

/// Write the simulation's energy grid as CSV. One row per grid point with
/// columns matching [`CSV_HEADER`]. Returns I/O errors verbatim.
pub fn write_grid_csv<W: Write>(writer: &mut W, result: &SimulationResult) -> io::Result<()> {
    writeln!(writer, "{CSV_HEADER}")?;
    for gp in &result.energy_grid {
        let pos = gp.position;
        let bands = gp.energy_bands;
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            safe_f32(pos.x),
            safe_f32(pos.y),
            safe_f32(pos.z),
            safe_f32(bands[0]),
            safe_f32(bands[1]),
            safe_f32(bands[2]),
            safe_f32(bands[3]),
            safe_f32(bands[4]),
            safe_f32(bands[5]),
            safe_f32(gp.energy),
        )?;
    }
    Ok(())
}

/// Write a human-readable text report covering: simulation config, per-listener
/// SPL across all 6 bands + broadband, and per-band RT60. Empty listener list
/// or all-None RT60 produces an explicit "—" placeholder rather than being
/// silently omitted.
pub fn write_text_report<W: Write>(
    writer: &mut W,
    config: &SimulationConfig,
    result: &SimulationResult,
) -> io::Result<()> {
    writeln!(writer, "Echomap Acoustic Simulation Report")?;
    writeln!(writer, "===================================")?;
    writeln!(writer)?;

    writeln!(writer, "Configuration")?;
    writeln!(writer, "-------------")?;
    writeln!(writer, "Ray count:        {}", config.ray_count)?;
    writeln!(writer, "Max bounces:      {}", config.max_bounces)?;
    writeln!(writer, "Energy threshold: {:.3e}", config.energy_threshold)?;
    writeln!(writer, "Grid resolution:  {} m", config.grid_resolution)?;
    writeln!(writer)?;

    writeln!(writer, "Grid Summary")?;
    writeln!(writer, "------------")?;
    writeln!(writer, "Grid points:      {}", result.energy_grid.len())?;
    writeln!(writer, "Max broadband:    {:.4e}", result.max_energy)?;
    writeln!(writer)?;

    // ---- Listener SPL table ----
    writeln!(writer, "Listener SPL (dB)")?;
    writeln!(writer, "-----------------")?;
    if result.listener_captures.is_empty() {
        writeln!(writer, "(no listeners in scene)")?;
    } else {
        writeln!(
            writer,
            "{:<24}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}",
            "Name", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "Broadband"
        )?;
        for cap in &result.listener_captures {
            write!(writer, "{:<24}", cap.name)?;
            for s in &cap.spl_bands {
                match s {
                    Some(v) => write!(writer, "  {v:8.1}")?,
                    None => write!(writer, "  {:>8}", "—")?,
                }
            }
            match cap.broadband_spl {
                Some(v) => writeln!(writer, "  {v:9.1}")?,
                None => writeln!(writer, "  {:>9}", "—")?,
            }
        }
    }
    writeln!(writer)?;

    // ---- RT60 table (room-wide, per band) ----
    writeln!(writer, "Reverberation Time RT60 (s)")?;
    writeln!(writer, "---------------------------")?;
    writeln!(
        writer,
        "{:<10}{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Band", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz"
    )?;
    write!(writer, "{:<10}", "RT60")?;
    for rt in &result.rt60_bands {
        match rt {
            Some(v) => write!(writer, "{v:8.3}  ")?,
            None => write!(writer, "{:>8}  ", "—")?,
        }
    }
    writeln!(writer)?;

    Ok(())
}

/// Export both CSV (energy grid) and TXT (human report) to the given paths.
/// Returns `ExportError::NoResults` if `result` is `None` so the UI can
/// surface a clean message — never panic on missing data.
pub fn export_simulation(
    csv_path: impl AsRef<Path>,
    txt_path: impl AsRef<Path>,
    config: &SimulationConfig,
    result: Option<&SimulationResult>,
) -> Result<(), ExportError> {
    let result = result.ok_or(ExportError::NoResults)?;
    let mut csv_file = File::create(csv_path).map_err(ExportError::Io)?;
    write_grid_csv(&mut csv_file, result).map_err(ExportError::Io)?;
    let mut txt_file = File::create(txt_path).map_err(ExportError::Io)?;
    write_text_report(&mut txt_file, config, result).map_err(ExportError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::ray::BAND_COUNT;
    use crate::acoustics::simulation::{
        GridPoint, ListenerCapture, SimulationConfig, SimulationResult,
    };
    use glam::Vec3;

    fn small_result() -> SimulationResult {
        let mut r = SimulationResult::default();
        r.energy_grid.push(GridPoint {
            position: Vec3::new(0.0, 0.0, 0.0),
            energy_bands: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            energy: 3.5,
        });
        r.energy_grid.push(GridPoint {
            position: Vec3::new(1.0, -0.25, 2.5),
            energy_bands: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            energy: 1.75,
        });
        r.energy_grid.push(GridPoint {
            position: Vec3::new(3.14, 2.71, 1.41),
            // Include a non-finite band value to confirm safe_f32 sanitises it.
            energy_bands: [f32::NAN, 0.0, 0.1, 0.2, 0.3, 0.4],
            energy: 0.1,
        });
        r
    }

    fn run_export(result: &SimulationResult) -> String {
        let mut buf: Vec<u8> = Vec::new();
        write_grid_csv(&mut buf, result).expect("write should succeed for in-memory buffer");
        String::from_utf8(buf).expect("CSV must be valid UTF-8")
    }

    #[test]
    fn csv_header() {
        let s = run_export(&small_result());
        let first_line = s.lines().next().expect("CSV has at least a header line");
        assert_eq!(
            first_line,
            "x,y,z,energy_125hz,energy_250hz,energy_500hz,energy_1khz,energy_2khz,energy_4khz,broadband",
            "header must match the deliverable schema byte-for-byte"
        );
    }

    #[test]
    fn csv_row_count() {
        let r = small_result();
        let s = run_export(&r);
        let total = s.lines().count();
        assert_eq!(
            total,
            r.energy_grid.len() + 1,
            "row count must be grid_points + 1 (header)"
        );
    }

    #[test]
    fn csv_parseable() {
        let s = run_export(&small_result());
        for (lineno, line) in s.lines().enumerate().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                10,
                "row {lineno} should have 10 columns (x,y,z,6 bands,broadband), got {}: {line}",
                fields.len()
            );
            for (i, f) in fields.iter().enumerate() {
                let v: f32 = f.parse().unwrap_or_else(|e| {
                    panic!("row {lineno} col {i} = `{f}` should parse as f32, err: {e}")
                });
                assert!(
                    v.is_finite(),
                    "row {lineno} col {i} ({v}) must be finite (sanitised from any NaN/Inf)"
                );
            }
        }
    }

    fn report_result_with_listener() -> SimulationResult {
        let mut r = small_result();
        let mut spl_bands: [Option<f32>; BAND_COUNT] = [None; BAND_COUNT];
        for i in 0..BAND_COUNT {
            spl_bands[i] = Some(80.0 - i as f32 * 3.0);
        }
        r.listener_captures.push(ListenerCapture {
            name: "L1".into(),
            position: Vec3::new(1.0, 1.0, 1.0),
            capture_radius: 0.3,
            energy_bands: [1.0; BAND_COUNT],
            spl_bands,
            broadband_energy: 1.0,
            broadband_spl: Some(70.0),
        });
        // 1 kHz RT60 is set, others None so the table renders mixed values.
        r.rt60_bands[3] = Some(0.42);
        r
    }

    #[test]
    fn report_contains_config() {
        let cfg = SimulationConfig {
            ray_count: 12345,
            max_bounces: 17,
            energy_threshold: 1e-4,
            grid_resolution: 0.42,
        };
        let r = report_result_with_listener();
        let mut buf: Vec<u8> = Vec::new();
        write_text_report(&mut buf, &cfg, &r).unwrap();
        let s = String::from_utf8(buf).unwrap();

        // Each named config field must appear with its value.
        assert!(
            s.contains("Ray count:") && s.contains("12345"),
            "ray_count must be in report:\n{s}"
        );
        assert!(
            s.contains("Max bounces:") && s.contains("17"),
            "max_bounces must be in report:\n{s}"
        );
        assert!(
            s.contains("Grid resolution:") && s.contains("0.42"),
            "grid_resolution must be in report:\n{s}"
        );
    }

    #[test]
    fn report_contains_listener_spl() {
        let cfg = SimulationConfig::default();
        let r = report_result_with_listener();
        let mut buf: Vec<u8> = Vec::new();
        write_text_report(&mut buf, &cfg, &r).unwrap();
        let s = String::from_utf8(buf).unwrap();

        // Listener name, header row, and broadband value should all be present.
        assert!(s.contains("L1"), "listener name must appear:\n{s}");
        assert!(
            s.contains("125Hz") && s.contains("4kHz") && s.contains("Broadband"),
            "SPL table header must include band labels and Broadband:\n{s}"
        );
        // Broadband SPL of 70.0 should appear as "70.0" somewhere on L1's row.
        assert!(s.contains("70.0"), "L1 broadband SPL must appear:\n{s}");
        // RT60 row label.
        assert!(
            s.contains("RT60"),
            "RT60 table must be present in report:\n{s}"
        );
    }

    #[test]
    fn export_no_results_error() {
        // Use a real but unlikely-to-exist temp path so the test fails on
        // create-error from "no results" rather than file-system noise.
        let tmpdir = std::env::temp_dir();
        let csv = tmpdir.join("echomap_test_no_results.csv");
        let txt = tmpdir.join("echomap_test_no_results.txt");
        let cfg = SimulationConfig::default();
        let result = export_simulation(&csv, &txt, &cfg, None);
        match result {
            Err(ExportError::NoResults) => { /* expected */ }
            other => panic!("expected ExportError::NoResults, got {other:?}"),
        }
        // And NEITHER file should have been created — failure is total.
        assert!(
            !csv.exists() || std::fs::metadata(&csv).unwrap().len() == 0,
            "CSV must not be created (or be empty) when no results"
        );
        // Clean up if anything slipped through.
        let _ = std::fs::remove_file(&csv);
        let _ = std::fs::remove_file(&txt);
    }

    #[test]
    fn csv_broadband_column() {
        let r = small_result();
        let s = run_export(&r);
        for ((idx, line), gp) in s.lines().enumerate().skip(1).zip(r.energy_grid.iter()) {
            let fields: Vec<f32> = line
                .split(',')
                .map(|x| x.parse::<f32>().unwrap_or_else(|_| panic!("parse `{x}`")))
                .collect();
            // Final column (index 9) is broadband. x,y,z (0..2), 6 bands (3..8), broadband (9).
            let csv_bb = fields[9];
            let expected = safe_f32(gp.energy);
            assert!(
                (csv_bb - expected).abs() < 1e-6,
                "row {idx}: broadband column ({csv_bb}) should equal GridPoint.energy ({expected})"
            );
            // Also confirm the per-band columns match the sanitised band array.
            for b in 0..BAND_COUNT {
                let expected = safe_f32(gp.energy_bands[b]);
                let actual = fields[3 + b];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "row {idx} band {b}: column ({actual}) should equal sanitised band ({expected})"
                );
            }
        }
    }
}
