//! End-to-end tele-op demo (goal/011 D7).
//!
//! Drives sinusoidal actions against a freshly-launched headless server,
//! writes the trace to JSONL, then replays it via [`Player::replay`] and
//! verifies determinism. Exits 0 on full success.
//!
//! CLI:
//!   --record PATH      Where to write the JSONL trace (required).
//!   --steps N          Number of steps to drive (default 100).
//!   --addr URL         WS server address (default ws://127.0.0.1:19002).
//!   --robot-id ID      Robot to bind (default 0).
//!   --tolerance T      Replay drift tolerance (default 1e-2).

use std::path::PathBuf;
use std::process::ExitCode;

use echomap::robot::state::RobotAction;
use echomap::teleop::playback::Player;
use echomap::teleop::run_session;

struct Args {
    record: PathBuf,
    steps: u64,
    addr: String,
    robot_id: usize,
    tolerance: f32,
}

fn parse_args() -> Result<Args, String> {
    let mut record: Option<PathBuf> = None;
    let mut steps: u64 = 100;
    let mut addr = "ws://127.0.0.1:19002".to_string();
    let mut robot_id: usize = 0;
    let mut tolerance: f32 = 1e-2;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--record" => {
                record = Some(PathBuf::from(it.next().ok_or("--record requires a path")?));
            }
            "--steps" => {
                steps = it
                    .next()
                    .ok_or("--steps requires a count")?
                    .parse()
                    .map_err(|e| format!("invalid --steps: {}", e))?;
            }
            "--addr" => {
                addr = it.next().ok_or("--addr requires a URL")?;
            }
            "--robot-id" => {
                robot_id = it
                    .next()
                    .ok_or("--robot-id requires an integer")?
                    .parse()
                    .map_err(|e| format!("invalid --robot-id: {}", e))?;
            }
            "--tolerance" => {
                tolerance = it
                    .next()
                    .ok_or("--tolerance requires a number")?
                    .parse()
                    .map_err(|e| format!("invalid --tolerance: {}", e))?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }

    Ok(Args {
        record: record.ok_or("--record PATH is required")?,
        steps,
        addr,
        robot_id,
        tolerance,
    })
}

fn print_help() {
    eprintln!(
        "teleop_e2e_demo --record PATH [--steps N] [--addr ws://host:port] [--robot-id ID] [--tolerance T]"
    );
}

fn sinusoidal_agent(step: u64, num_motors: usize) -> RobotAction {
    let t = step as f32 * 0.05;
    let motor_velocities: Vec<f32> = (0..num_motors)
        .map(|m| (t + m as f32 * 0.3).sin())
        .collect();
    RobotAction {
        motor_velocities,
        gripper_commands: Vec::new(),
        base_velocity: [0.0, 0.0],
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}", e);
            print_help();
            return ExitCode::from(2);
        }
    };

    if let Some(parent) = args.record.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("error: create trace dir {:?}: {}", parent, e);
            return ExitCode::from(2);
        }
    }

    eprintln!(
        "[teleop_e2e_demo] record phase: addr={} robot_id={} steps={} path={:?}",
        args.addr, args.robot_id, args.steps, args.record
    );

    let record_result = run_session(
        &args.addr,
        args.robot_id,
        args.steps,
        |step, obs| sinusoidal_agent(step, obs.joint_positions.len().max(1)),
        &args.record,
    )
    .await;

    let frames = match record_result {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[teleop_e2e_demo] record FAILED: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!("[teleop_e2e_demo] record OK ({} frames)", frames);

    eprintln!(
        "[teleop_e2e_demo] replay phase: addr={} robot_id={} tolerance={}",
        args.addr, args.robot_id, args.tolerance
    );

    let report = match Player::replay(&args.addr, args.robot_id, &args.record, args.tolerance).await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[teleop_e2e_demo] replay transport FAILED: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "[teleop_e2e_demo] replay report: total={} matched={} max_drift={:.6} diverged_at={:?}",
        report.frames_total, report.frames_matched, report.max_drift, report.diverged_at
    );

    if report.passed() {
        eprintln!("[teleop_e2e_demo] OK — record + replay determinism within tolerance");
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "[teleop_e2e_demo] FAIL — diverged at frame {:?} (max_drift {:.6} > tol {})",
            report.diverged_at, report.max_drift, args.tolerance
        );
        ExitCode::from(1)
    }
}
