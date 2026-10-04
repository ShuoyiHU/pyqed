"""Click-run one configurable 1D chain; exact CBE is off by default."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import run_model_cli
from pyqed._letta_one_site_opt.benchmarks.condensed_models import MODEL_NAMES

# Edit MODEL for IDE click-run, or select --model on the command line.
MODEL = "ssh"
DEFAULT_SOLVERS = ("letta_one_site", "letta_two_site", "letta_cbe_strict")


def main(argv=None):
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--model", choices=MODEL_NAMES, default=MODEL)
    args, remaining = selection.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print("Select a chain with --model {" + ",".join(MODEL_NAMES) + "}.")
    return run_model_cli(args.model, "1d", [
        "--solvers", ",".join(DEFAULT_SOLVERS), *remaining,
    ])


if __name__ == "__main__":
    main()
