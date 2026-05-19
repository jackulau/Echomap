//! Result export — CSV (per-grid-point band energies) and text reports.
//!
//! Schema is fixed: the CSV header is exact and consumers parse positionally.
//! Changing it is a versioned data-contract change, not a refactor.

use std::io::{self, Write};

use crate::acoustics::simulation::SimulationResult;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::ray::BAND_COUNT;
    use crate::acoustics::simulation::{GridPoint, SimulationResult};
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
