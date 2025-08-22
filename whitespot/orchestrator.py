# ----------------------------
# Orchestrator & CLI
# ----------------------------

import logging

import numpy as np
import pandas as pd
from whitespot.dataloader import DataLoader
from whitespot.evaluator import Evaluator
from whitespot.recommender.alsrecommender import ALSRecommender
from whitespot.recommender.baserecommender import BaseRecommender
from whitespot.recommender.cosinerecommender import CosineRecommender
from whitespot.recommender.covisrecommender import CoVisRecommender
from whitespot.symbols import OUTPUTDIR


class Orchestrator:
    def __init__(self) -> None:
        self.loader = DataLoader()

    def run(
        self,
        engine: str = "baseline",
        nrows: int = 5000,
        top_n: int = 10,
        min_support: int = 3,
        covis_topk: int = 100,
        covis_score: str = "lift",
        als_factors: int = 64,
        als_iterations: int = 15,
        als_regularization: float = 0.01,
        als_alpha: float = 40.0,
        als_use_qty: bool = False,
    ) -> None:
        df = self.loader.read(nrows=nrows)

        # Diagnostics
        stats = (
            df.groupby("PartID")
            .agg(
                customers=("CustomerID", "nunique"),
                baskets=("OrderDocID", "nunique"),
                last_date=("OrderDocDate", "max"),
                total_qty=("QtyInBasket", "sum"),
            )
            .reset_index()
        )
        stats.to_csv(
            OUTPUTDIR / "item_stats.csv", index=False, sep=";", encoding="utf-8"
        )

        # Choose engine
        if engine == "cosine":
            model: BaseRecommender = CosineRecommender(min_support=min_support)
        elif engine == "covis":
            model = CoVisRecommender(
                topk_neighbors=covis_topk, score=covis_score, min_support=min_support
            )
        elif engine == "als":
            model = ALSRecommender(
                factors=als_factors,
                iterations=als_iterations,
                regularization=als_regularization,
                alpha=als_alpha,
                use_qty=als_use_qty,
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")

        model.fit(df)

        # Export recommendations
        rec_df = model.recommend_all(top_n=top_n)
        part1 = DataLoader.part_dict(df)
        rec_df["PartDesc"] = rec_df["PartID"].map(
            lambda pid: part1.loc[pid, "PartDesc1"] if pid in part1.index else ""
        )
        suffix = f"_{engine}"
        rec_df.to_csv(
            OUTPUTDIR / f"recommendations_per_customer{suffix}.csv",
            index=False,
            sep=";",
            encoding="utf-8",
        )

        # Evaluate
        train, test = Evaluator.temporal_split(df)
        hit, cov = Evaluator.hitrate_at_k(model.fit(train), train, test, k=top_n)
        meta = {
            "engine": engine,
            "n_customers": df["CustomerID"].nunique(),
            "n_items": df["PartID"].nunique(),
            "min_support": min_support,
            "top_n": top_n,
            "hit_rate_at_k": hit,
            "coverage": cov,
            "covis_topk": covis_topk if engine == "covis" else np.nan,
            "als_factors": als_factors if engine == "als" else np.nan,
        }
        pd.Series(meta).to_csv(
            OUTPUTDIR / f"run_metadata{suffix}.csv", sep=";", header=False
        )
        logging.info("Saved outputs to %s", OUTPUTDIR)
        logging.info(
            "[%s] HitRate@%d = %.3f | Coverage = %.3f", engine, top_n, hit, cov
        )
