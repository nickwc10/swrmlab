"""The ``swarmsim`` command: fly the demo, plot a replay."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

DEFAULT_SCENARIO = Path("scenarios/two_outposts.yaml")
_SEED_LIMIT = 2**63  # what a replay header accepts


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        """One line and exit code 2, without the usage text."""
        self.exit(2, f"swarmsim: error: {message}\n")


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if not 0 <= value < _SEED_LIMIT:
        raise argparse.ArgumentTypeError("must be a whole number from 0 to 2**63 - 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="swarmsim", description="Autonomous space battle simulator.")
    commands = parser.add_subparsers(dest="command", metavar="command", required=True)
    demo = commands.add_parser("demo", help="fly a scripted crossing and save its replay")
    demo.add_argument("--seed", type=_seed, required=True, help="whole number, 0 or more")
    demo.add_argument("--out", type=Path, required=True, help="replay file to write")
    demo.add_argument(
        "--scenario", type=Path, default=DEFAULT_SCENARIO, help="default: %(default)s"
    )
    plot = commands.add_parser("plot", help="draw a replay as a PNG picture")
    plot.add_argument("replay", type=Path, help="replay file to read")
    plot.add_argument("--out", type=Path, required=True, help="PNG file to write")
    return parser


def _demo(args: argparse.Namespace) -> None:
    from swarmsim.demo import run_demo
    from swarmsim.replay import write
    from swarmsim.scenario import load

    try:
        scenario = load(args.scenario)
    except ValueError as err:
        raise ValueError(f"scenario {args.scenario}: {_one_line(err)}") from None
    _refuse_input(args.out, args.scenario, "scenario")
    world = run_demo(scenario, args.seed)
    _prepare(args.out)
    write(args.out, world, scenario, args.seed)


def _plot(args: argparse.Namespace) -> None:
    from swarmsim.plot import plot
    from swarmsim.replay import ReplayError, read

    try:
        replay = read(args.replay)
    except ReplayError as err:
        raise ValueError(f"replay {args.replay}: {_one_line(err)}") from None
    _refuse_input(args.out, args.replay, "replay")
    _prepare(args.out)
    plot(replay, args.out)


def _refuse_input(out: Path, source: Path, name: str) -> None:
    """Refuse an ``out`` that is the file just read, however either path is spelled."""
    if out.exists() and out.samefile(source):
        raise ValueError(f"--out is the {name} itself: {out}")


def _prepare(out: Path) -> None:
    """Make the folder ``out`` goes in, and refuse a folder as the file."""
    if out.is_dir():
        raise IsADirectoryError(f"{out} is a folder, not a file")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise OSError(f"cannot make the folder for {out}: {err.strerror}") from None


def _one_line(err: Exception) -> str:
    if isinstance(err, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in e['loc']) or 'scenario'}: {e['msg']}"
            for e in err.errors(include_url=False)
        ]
        return "; ".join(parts)
    return " ".join(str(err).split()) or type(err).__name__


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        (_demo if args.command == "demo" else _plot)(args)
    except (OSError, ValueError, RecursionError) as err:
        print(f"swarmsim: error: {_one_line(err)}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0
