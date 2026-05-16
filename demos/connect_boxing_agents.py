#!/usr/bin/env python3
"""Connect two boxing agents to an already-running EchoMap GUI server.

Usage:
    1. Launch the EchoMap GUI:  cargo run
    2. Click "Start Boxing Match" in the Agent Server panel
    3. Run this script:  python3 demos/connect_boxing_agents.py
    4. (Optional) --mode ollama for LLM agents
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from echomap_client.env import EchoMapEnv


def create_agents(mode, model):
    from echomap_client.agents import HeuristicBoxingAgent

    if mode == "ollama":
        from echomap_client.ollama_agent import OllamaBoxingAgent
        m = model or "llama3.2"
        return (
            OllamaBoxingAgent(model=m, name=f"IronFist ({m})"),
            OllamaBoxingAgent(model=m, name=f"ThunderBot ({m})"),
        )
    elif mode == "llm":
        from echomap_client.llm_agent import LLMBoxingAgent
        m = model or "claude-haiku-4-5-20251001"
        return (
            LLMBoxingAgent(model=m, name="IronFist (Claude)"),
            LLMBoxingAgent(model=m, name="ThunderBot (Claude)"),
        )
    else:
        return (
            HeuristicBoxingAgent(name="IronFist-3000", trash_talk_chance=0.05),
            HeuristicBoxingAgent(name="ThunderBot-X", trash_talk_chance=0.05),
        )


def main():
    parser = argparse.ArgumentParser(description="Connect agents to running GUI server")
    parser.add_argument("--mode", choices=["heuristic", "ollama", "llm"], default="heuristic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--port", type=int, default=9002, help="WS port of running server")
    args = parser.parse_args()

    agent_a, agent_b = create_agents(args.mode, args.model)
    print(f"Connecting {agent_a.name} vs {agent_b.name} to localhost:{args.port}...")

    env_a = EchoMapEnv(host="localhost", port=args.port, robot_id=0)
    env_b = EchoMapEnv(host="localhost", port=args.port, robot_id=1)

    env_a.connect()
    env_b.connect()
    print("Both fighters connected!")

    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()

    def phase_cat(phase):
        for prefix in ("round_end", "countdown"):
            if phase.startswith(prefix):
                return prefix
        return phase

    last_cat = ""
    step = 0

    try:
        while True:
            action_a, msg_a = agent_a.decide(obs_a, info_a)
            action_b, msg_b = agent_b.decide(obs_b, info_b)

            if msg_a:
                try:
                    env_a.send_message(1, msg_a)
                    print(f"  {agent_a.name}: \"{msg_a}\"")
                except Exception:
                    pass
            if msg_b:
                try:
                    env_b.send_message(0, msg_b)
                    print(f"  {agent_b.name}: \"{msg_b}\"")
                except Exception:
                    pass

            obs_a, _, done_a, info_a = env_a.step(action_a)
            obs_b, _, done_b, info_b = env_b.step(action_b)
            step += 1

            ms = (info_b or {}).get("match_state") or (info_a or {}).get("match_state")
            phase = ms.get("phase", "") if ms else ""
            cat = phase_cat(phase)

            if cat != last_cat:
                if cat == "fighting":
                    rnd = ms.get("current_round", "?")
                    print(f"\n  >>> ROUND {rnd} - FIGHT! <<<\n")
                elif cat == "round_end":
                    sa = ms.get("total_score_a", 0)
                    sb = ms.get("total_score_b", 0)
                    print(f"\n  --- Round over! Score: {sa}-{sb} ---\n")
                elif cat == "match_end":
                    break
            last_cat = cat

            if done_a or done_b:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        final_ms = (info_b or {}).get("match_state") or (info_a or {}).get("match_state")
        if final_ms:
            sa = final_ms.get("total_score_a", 0)
            sb = final_ms.get("total_score_b", 0)
            print(f"\nFinal Score: {sa} - {sb}  ({step} steps)")
            if sa > sb:
                print(f"WINNER: {agent_a.name}")
            elif sb > sa:
                print(f"WINNER: {agent_b.name}")
            else:
                print("DRAW!")

        env_a.close()
        env_b.close()


if __name__ == "__main__":
    main()
