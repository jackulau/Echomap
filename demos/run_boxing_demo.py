#!/usr/bin/env python3
"""Run a full boxing match demo: start headless server, run agents, capture output.

Usage:
    python3 demos/run_boxing_demo.py                     # heuristic agents
    python3 demos/run_boxing_demo.py --mode ollama       # local Ollama (llama3.2)
    python3 demos/run_boxing_demo.py --mode ollama --model qwen2.5:0.5b
    python3 demos/run_boxing_demo.py --mode llm          # Claude API
"""

import argparse
import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.path.join(REPO_ROOT, "target", "debug", "echomap_server")
DEMO_OUTPUT = os.path.join(REPO_ROOT, "demos", "boxing_match_demo.txt")

WS_PORT = 19002
TCP_PORT = 19001
ROUND_DURATION = 5  # overridden for LLM modes below
NUM_ROUNDS = 3


def build_server():
    print("Building headless server...")
    result = subprocess.run(
        ["cargo", "build", "--bin", "echomap_server"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Build complete.")


def start_server(round_duration, num_rounds):
    env = os.environ.copy()
    env["TCP_PORT"] = str(TCP_PORT)
    env["WS_PORT"] = str(WS_PORT)
    env["ROUND_DURATION"] = str(round_duration)
    env["NUM_ROUNDS"] = str(num_rounds)
    proc = subprocess.Popen(
        [SERVER_BIN],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    for _ in range(50):
        time.sleep(0.1)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            print(f"Server exited early:\n{stderr}", file=sys.stderr)
            sys.exit(1)
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(("localhost", WS_PORT))
            s.close()
            return proc
        except (ConnectionRefusedError, OSError):
            continue
    print("Server did not start in time", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def out(lines, text):
    print(text)
    lines.append(text)


def create_demo_agents(mode, model):
    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    from echomap_client.agents import HeuristicBoxingAgent

    if mode == "ollama":
        from echomap_client.ollama_agent import OllamaBoxingAgent
        m = model or "llama3.2"
        return (
            OllamaBoxingAgent(model=m, name=f"IronFist ({m})"),
            OllamaBoxingAgent(model=m, name=f"ThunderBot ({m})"),
            f"Ollama ({m})",
        )
    elif mode == "llm":
        from echomap_client.llm_agent import LLMBoxingAgent
        m = model or "claude-haiku-4-5-20251001"
        return (
            LLMBoxingAgent(model=m, name="IronFist (Claude)"),
            LLMBoxingAgent(model=m, name="ThunderBot (Claude)"),
            f"Claude API ({m})",
        )
    else:
        return (
            HeuristicBoxingAgent(name="IronFist-3000", trash_talk_chance=0.05),
            HeuristicBoxingAgent(name="ThunderBot-X", trash_talk_chance=0.05),
            "Heuristic (rule-based)",
        )


def run_match(mode="heuristic", model=None, round_duration=5, num_rounds=3):
    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    from echomap_client.env import EchoMapEnv
    from echomap_client.commentary import MatchCommentary

    agent_a, agent_b, mode_desc = create_demo_agents(mode, model)
    commentary = MatchCommentary(use_llm=False)

    lines = []
    out(lines, "=" * 60)
    out(lines, "  ECHOMAP AI BOXING MATCH DEMO")
    out(lines, "=" * 60)
    out(lines, "")
    out(lines, f"  Fighter A: {agent_a.name}")
    out(lines, f"  Fighter B: {agent_b.name}")
    out(lines, f"  Mode: {mode_desc}")
    out(lines, f"  Server: localhost:{WS_PORT}")
    out(lines, f"  Rounds: {num_rounds} x {round_duration}s")
    out(lines, "")
    out(lines, "-" * 60)
    out(lines, "  FIGHT!")
    out(lines, "-" * 60)
    out(lines, "")

    env_a = EchoMapEnv(host="localhost", port=WS_PORT, robot_id=0)
    env_b = EchoMapEnv(host="localhost", port=WS_PORT, robot_id=1)

    env_a.connect()
    env_b.connect()
    out(lines, "  [Both fighters enter the ring]")

    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()

    stats = {"steps": 0, "messages_a": 0, "messages_b": 0, "rounds_completed": 0}
    last_phase_cat = ""
    commentary_log = []

    def phase_category(phase):
        for prefix in ("round_end", "countdown"):
            if phase.startswith(prefix):
                return prefix
        return phase

    for step in range(10000):
        action_a, msg_a = agent_a.decide(obs_a, info_a)
        action_b, msg_b = agent_b.decide(obs_b, info_b)

        if msg_a:
            try:
                env_a.send_message(1, msg_a)
                stats["messages_a"] += 1
                out(lines, f"  {agent_a.name}: \"{msg_a}\"")
            except Exception:
                pass
        if msg_b:
            try:
                env_b.send_message(0, msg_b)
                stats["messages_b"] += 1
                out(lines, f"  {agent_b.name}: \"{msg_b}\"")
            except Exception:
                pass

        obs_a, _, done_a, info_a = env_a.step(action_a)
        obs_b, _, done_b, info_b = env_b.step(action_b)
        stats["steps"] += 1

        for h in (info_a or {}).get("hit_events", []) + (info_b or {}).get("hit_events", []):
            stats.setdefault("total_hits", 0)
            stats["total_hits"] += 1

        ms_a = info_a.get("match_state") if info_a else None
        ms_b = info_b.get("match_state") if info_b else None
        match_state = ms_b or ms_a
        current_phase = match_state.get("phase", "") if match_state else ""
        current_cat = phase_category(current_phase)

        if current_cat != last_phase_cat:
            if current_cat == "fighting":
                round_num = match_state.get("current_round", "?")
                out(lines, f"\n  >>> ROUND {round_num} - FIGHT! <<<\n")
            elif current_cat == "round_end":
                stats["rounds_completed"] += 1
                sa = match_state.get("total_score_a", 0)
                sb = match_state.get("total_score_b", 0)
                out(lines, f"\n  --- Round over! Score: {sa}-{sb} ---")
                summary = commentary.generate_round_summary(match_state)
                commentary_log.append(summary)
                out(lines, f"  [Commentary] {summary}\n")
            elif current_cat == "match_end":
                break

        hits_a = info_a.get("hit_events", []) if info_a else []
        for hit in hits_a:
            zone = hit.get("zone", "body")
            force = hit.get("force", 0)
            if force > 5:
                out(lines, f"  ** BIG HIT to {zone}! (force: {force:.1f}) **")

        last_phase_cat = current_cat

        if done_a or done_b:
            break

    # Final summary — use latest state (agent B steps second, has latest)
    final_state = (info_b.get("match_state") if info_b else None) or \
                  (info_a.get("match_state") if info_a else None)
    if final_state:
        match_summary = commentary.generate_match_summary(final_state)
        commentary_log.append(match_summary)

    env_a.close()
    env_b.close()

    out(lines, "")
    out(lines, "=" * 60)
    out(lines, "  Match Result")
    out(lines, "=" * 60)
    out(lines, "")

    if final_state:
        sa = final_state.get("total_score_a", 0)
        sb = final_state.get("total_score_b", 0)
        if sa > sb:
            out(lines, f"  WINNER: {agent_a.name}!")
        elif sb > sa:
            out(lines, f"  WINNER: {agent_b.name}!")
        else:
            out(lines, "  RESULT: DRAW!")
        out(lines, f"  Score: {sa} - {sb}")

    out(lines, f"  Total Steps: {stats['steps']}")
    out(lines, f"  Trash Talk: {agent_a.name}={stats['messages_a']}, "
               f"{agent_b.name}={stats['messages_b']}")
    total_rounds = final_state.get("current_round", stats["rounds_completed"]) if final_state else stats["rounds_completed"]
    out(lines, f"  Rounds: {total_rounds}")
    out(lines, f"  Hit Events: {stats.get('total_hits', 0)}")
    out(lines, "")

    if commentary_log:
        out(lines, "-" * 60)
        out(lines, "  COMMENTARY")
        out(lines, "-" * 60)
        for c in commentary_log:
            out(lines, f"  {c}")
        out(lines, "")

    out(lines, "=" * 60)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Run a boxing match demo")
    parser.add_argument("--mode", choices=["heuristic", "ollama", "llm"], default="heuristic")
    parser.add_argument("--model", default=None, help="Model name for LLM/Ollama")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--round-duration", type=int, default=None, help="Seconds per round")
    args = parser.parse_args()

    is_llm = args.mode in ("ollama", "llm")
    round_duration = args.round_duration or (2 if is_llm else 5)
    num_rounds = args.rounds or (1 if is_llm else 3)

    build_server()

    print("\nStarting headless boxing server...")
    server_proc = start_server(round_duration, num_rounds)
    print(f"Server ready on TCP:{TCP_PORT} WS:{WS_PORT}\n")

    try:
        lines = run_match(mode=args.mode, model=args.model,
                          round_duration=round_duration, num_rounds=num_rounds)
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("\nServer stopped.")

    with open(DEMO_OUTPUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nDemo saved to: {DEMO_OUTPUT}")


if __name__ == "__main__":
    main()
