pub mod analytical;

#[macro_export]
macro_rules! assert_relative_eq {
    ($actual:expr, $expected:expr, $tol:expr) => {
        let a = $actual as f64;
        let e = $expected as f64;
        let rel_err = if e.abs() < 1e-15 {
            a.abs()
        } else {
            ((a - e) / e).abs()
        };
        assert!(
            rel_err <= $tol,
            "relative error {:.6} exceeds tolerance {:.6} (actual={}, expected={})",
            rel_err,
            $tol,
            a,
            e
        );
    };
}
