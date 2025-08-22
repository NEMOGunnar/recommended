"""
Whitespot Analysis (class-based)
--------------------------------
- Three models:
  * CosineRecommender (baseline item-item CF)
  * CoVisRecommender (co-visitation Top-K)
  * ALSRecommender (implicit ALS)
- Orchestrator handles I/O, evaluation, CLI.
"""

import logging
import argparse

from whitespot.orchestrator import Orchestrator


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whitespot analysis (class-based)")
    p.add_argument("--nrows", type=int, default=5000)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--engine", choices=["cosine", "covis", "als"], default="cosine")
    p.add_argument("--covis-topk", type=int, default=100)
    p.add_argument(
        "--covis-score", choices=["lift", "count", "jaccard"], default="lift"
    )
    p.add_argument("--als-factors", type=int, default=64)
    p.add_argument("--als-iterations", type=int, default=15)
    p.add_argument("--als-regularization", type=float, default=0.01)
    p.add_argument("--als-alpha", type=float, default=40.0)
    p.add_argument("--als-use-qty", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    orch = Orchestrator()
    orch.run(
        engine=args.engine,
        nrows=args.nrows,
        top_n=args.top_n,
        min_support=args.min_support,
        covis_topk=args.covis_topk,
        covis_score=args.covis_score,
        als_factors=args.als_factors,
        als_iterations=args.als_iterations,
        als_regularization=args.als_regularization,
        als_alpha=args.als_alpha,
        als_use_qty=args.als_use_qty,
    )


if __name__ == "__main__":
    main()
