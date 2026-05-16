use echomap::agent::{AgentServerConfig, HeadlessServer};
use echomap::robot::boxing::BoxingMatchConfig;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    env_logger::init();

    let tcp_port: u16 = env_parse("TCP_PORT", 9001);
    let ws_port: u16 = env_parse("WS_PORT", 9002);
    let round_duration: f32 = env_parse("ROUND_DURATION", 180.0);
    let num_rounds: u8 = env_parse("NUM_ROUNDS", 3);

    let server_config = AgentServerConfig {
        tcp_port,
        ws_port,
        max_connections: 16,
        enabled: true,
    };

    let boxing_config = BoxingMatchConfig {
        round_duration,
        num_rounds,
        ..BoxingMatchConfig::default()
    };

    eprintln!("Starting headless boxing server...");
    eprintln!("  Rounds: {} x {:.0}s", num_rounds, round_duration);
    let server = HeadlessServer::start_boxing_with(server_config, boxing_config);
    let status = server.status();
    eprintln!(
        "Boxing server ready — TCP:{} WS:{}",
        status.tcp_port, status.ws_port
    );
    eprintln!("Press Ctrl+C to stop.");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("failed to set Ctrl+C handler");

    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    eprintln!("Shutting down...");
    server.stop();
}
