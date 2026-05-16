"""CLI entry point for running boxing matches + plugin management."""

import argparse
import sys


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run an AI boxing match in EchoMap simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "ollama", "mixed"],
        default="heuristic",
        help="Agent mode: heuristic (rule-based), llm (Claude API), ollama (local Ollama), mixed (LLM vs heuristic)",
    )
    parser.add_argument("--model", default=None, help="Model name (e.g. llama3.2, claude-haiku-4-5-20251001)")
    parser.add_argument("--host", default="localhost", help="Simulation server host")
    parser.add_argument("--port", type=int, default=9002, help="Simulation server port")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--verbose", action="store_true", help="Print match progress")
    return parser.parse_args(args)


def list_plugins_cmd(argv):
    """Print discovered plugins. `argv` reserved for future flags (e.g. --json)."""
    from .plugins import load_all
    parser = argparse.ArgumentParser(
        prog="echomap_client.cli list-plugins",
        description="List installed EchoMap plugins (agents / sensors / scenarios / visualizations / hardware)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    opts = parser.parse_args(argv)
    reg = load_all()
    if opts.json:
        import json
        payload = {
            group: {name: f"{getattr(obj, '__module__', '?')}:{getattr(obj, '__qualname__', repr(obj))}"
                    for name, obj in reg.get(group).items()}
            for group in [
                "echomap.plugins.agents",
                "echomap.plugins.sensors",
                "echomap.plugins.scenarios",
                "echomap.plugins.visualizations",
                "echomap.plugins.hardware",
            ]
        }
        payload["_errors"] = reg.errors
        print(json.dumps(payload, indent=2))
    else:
        print(reg.summary())
    return 0


def create_agents(mode, model=None):
    from .agents import HeuristicBoxingAgent

    if mode == "heuristic":
        return (
            HeuristicBoxingAgent(name="Robot A", trash_talk_chance=0.1),
            HeuristicBoxingAgent(name="Robot B", trash_talk_chance=0.1),
        )
    elif mode == "llm":
        from .llm_agent import LLMBoxingAgent
        kwargs = {}
        if model:
            kwargs["model"] = model
        return (
            LLMBoxingAgent(name="LLM-A", **kwargs),
            LLMBoxingAgent(name="LLM-B", **kwargs),
        )
    elif mode == "ollama":
        from .ollama_agent import OllamaBoxingAgent
        m = model or "llama3.2"
        return (
            OllamaBoxingAgent(model=m, name=f"Ollama-A ({m})"),
            OllamaBoxingAgent(model=m, name=f"Ollama-B ({m})"),
        )
    elif mode == "mixed":
        from .llm_agent import LLMBoxingAgent
        return (
            LLMBoxingAgent(name="LLM"),
            HeuristicBoxingAgent(name="Heuristic", trash_talk_chance=0.15),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main(args=None):
    raw = sys.argv[1:] if args is None else list(args)
    # Subcommand dispatch — first positional decides. Backwards compat: any
    # invocation without a recognized subcommand falls through to the match
    # runner.
    if raw and raw[0] == "list-plugins":
        return list_plugins_cmd(raw[1:])

    parsed = parse_args(args)

    if parsed.rounds < 1:
        print("Error: --rounds must be at least 1", file=sys.stderr)
        sys.exit(1)

    agent_a, agent_b = create_agents(parsed.mode, parsed.model)

    print(f"Boxing Match: {agent_a.name} vs {agent_b.name}")
    print(f"Mode: {parsed.mode} | Server: {parsed.host}:{parsed.port}")
    print(f"Rounds: {parsed.rounds}")
    print()

    from .runner import BoxingMatchRunner

    runner = BoxingMatchRunner(
        agent_a, agent_b,
        host=parsed.host,
        port=parsed.port,
        verbose=parsed.verbose,
    )

    try:
        result = runner.run()
    except Exception as e:
        print(f"Match error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=== Match Result ===")
    if result["winner"]:
        winner_name = agent_a.name if result["winner"] == "a" else agent_b.name
        print(f"Winner: {winner_name}")
    else:
        print("Result: Draw!")
    print(f"Score: {result['scores']['a']}-{result['scores']['b']}")
    print(f"Steps: {result['stats']['steps']}")
    print(f"Messages: A={result['stats']['messages_a']}, B={result['stats']['messages_b']}")

    if result["commentary"]:
        print()
        print("=== Commentary ===")
        for line in result["commentary"]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
