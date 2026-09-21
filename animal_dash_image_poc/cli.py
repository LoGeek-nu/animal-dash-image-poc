"""CLI entrypoint: preprocess / status / run subcommands."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from . import gemini_status, pipeline


def cmd_preprocess(args: argparse.Namespace) -> None:
    pipeline.preprocess_file(args.input, args.output, debug_dir=args.debug)
    print(f"wrote {args.output}" + (f" (debug: {args.debug})" if args.debug else ""))


def cmd_status(args: argparse.Namespace) -> None:
    result = gemini_status.generate_status(args.input, api_key=args.api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    result = pipeline.run_full_pipeline(
        args.input, args.output, api_key=args.api_key, debug_dir=args.debug
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}.png and {args.output}.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="animal-dash-image-poc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pre = subparsers.add_parser("preprocess", help="Run OpenCV preprocessing only")
    p_pre.add_argument("input", help="Path to input photo (jpg/png)")
    p_pre.add_argument("-o", "--output", required=True, help="Path to output transparent PNG")
    p_pre.add_argument("--debug", help="Directory to dump intermediate step images")
    p_pre.set_defaults(func=cmd_preprocess)

    p_status = subparsers.add_parser("status", help="Run Gemini status judgment on a preprocessed PNG")
    p_status.add_argument("input", help="Path to preprocessed transparent PNG")
    p_status.add_argument("--api-key", help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    p_status.set_defaults(func=cmd_status)

    p_run = subparsers.add_parser("run", help="Run preprocessing + Gemini status end-to-end")
    p_run.add_argument("input", help="Path to input photo (jpg/png)")
    p_run.add_argument("-o", "--output", required=True, help="Output prefix (writes <prefix>.png and <prefix>.json)")
    p_run.add_argument("--debug", help="Directory to dump intermediate step images")
    p_run.add_argument("--api-key", help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
