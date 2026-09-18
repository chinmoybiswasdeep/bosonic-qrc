"""CLI for the corrected finite-size EOC workflow."""

import argparse
from pathlib import Path

from .floquet_reservoir import MBFloquetConfig, MBFloquetReservoir
from .workflow import merge_chunks, run_workflow


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--chunk-index", type=int, default=0)
    run.add_argument("--chunk-count", type=int, default=1)
    run.add_argument("--closed-result", type=Path)
    run.add_argument("--resume", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "merge":
        manifest = merge_chunks(args.inputs, args.output)
    else:
        manifest = run_workflow(
            args.config,
            args.output,
            MBFloquetReservoir,
            MBFloquetConfig,
            "cv",
            chunk_index=args.chunk_index,
            chunk_count=args.chunk_count,
            closed_result=args.closed_result,
            resume=args.resume,
        )
    print(
        {
            "status": manifest["gate"]["status"],
            "eoc_claim": manifest["gate"]["edge_of_many_body_quantum_chaos_claim"],
        }
    )


if __name__ == "__main__":
    main()
