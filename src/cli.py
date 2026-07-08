"""Command-line entry point for the LLM stress test suite.

Usage:
    python src/cli.py
    python src/cli.py --models Qwen/Qwen2-0.5B --categories jailbreak
    python src/cli.py --config path/to/config.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluator import run_stress_test

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path) -> dict:
    """Load the YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run adversarial stress tests against one or more LLMs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", help="Override the model list from the config")
    parser.add_argument(
        "--categories", nargs="+", choices=["hallucination", "jailbreak"],
        help="Override the evaluation categories from the config",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)

    models      = args.models or config["models"]["list"]
    categories  = args.categories or config["evaluation"]["categories"]
    gen_config  = config.get("generation", {})
    reports_dir = DEFAULT_CONFIG_PATH.parent.parent / config["paths"]["reports_dir"]

    summary = []
    for model_name in models:
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")

        def log(step, total, message):
            print(f"  [{step}/{total}] {message}")

        try:
            result = run_stress_test(
                model_name,
                categories,
                generation_config=gen_config,
                reports_dir=reports_dir,
                progress_callback=log,
            )
            print(
                f"Robustness: {result['robustness_pct']}%  "
                f"(safe={result['safe']} unsafe={result['unsafe']} "
                f"unclear={result['unclear']} errors={result['errors']})"
            )
            summary.append((model_name, result["robustness_pct"]))
        except Exception as exc:
            print(f"Failed to evaluate {model_name}: {exc}")
            summary.append((model_name, None))

    print(f"\n{'-' * 60}\nSummary\n{'-' * 60}")
    for model_name, pct in summary:
        status = f"{pct}%" if pct is not None else "FAILED"
        print(f"  {model_name}: {status}")


if __name__ == "__main__":
    main()