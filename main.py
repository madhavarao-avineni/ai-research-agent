"""
main.py
=======
Headless CLI entry point for the AI Research Assistant.

Usage:
    python main.py --question "Your research question here"
    python main.py --question "..." --max-iterations 2 --output report.md

Runs the full LangGraph workflow and writes the Markdown report to disk.
"""

from __future__ import annotations

import argparse
import sys

from graph.research_graph import run_research
from observability.logging_config import configure_logging, get_logger

log = get_logger("main")

DEFAULT_QUESTION = (
    "Will generative AI significantly reduce the demand for software developers "
    "over the next five years?"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Multi-Agent Research Assistant")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION, help="Research question")
    parser.add_argument("--max-iterations", "-m", type=int, default=None, help="Max extra research loops")
    parser.add_argument("--output", "-o", default="research_report.md", help="Output Markdown path")
    args = parser.parse_args(argv)

    configure_logging()
    log.info("Running research for: %r", args.question)

    final_state = run_research(args.question, max_iterations=args.max_iterations)
    report = final_state.get("final_report")

    if not report or not report.markdown:
        log.error("No report was produced.")
        print("ERROR: no report produced. Check logs / configuration.")
        return 1

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report.markdown)
    except OSError as exc:
        log.error("Failed to write report: %s", exc)
        print(report.markdown)
        return 1

    print(f"\n[OK] Report written to {args.output}")
    print(f"   Sources: {len(final_state.get('sources', []))}")
    print(f"   Iterations: {final_state.get('research_iteration', 0)}")
    print(f"   Agents run: {len(final_state.get('traces', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
