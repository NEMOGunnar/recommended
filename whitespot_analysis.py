"""
Whitespot Analysis (class-based)
--------------------------------
- English comments and outputs
- Three models:
  * CosineRecommender (baseline item-item CF)
  * CoVisRecommender (co-visitation Top-K)
  * ALSRecommender (implicit ALS)
- Orchestrator handles I/O, evaluation, CLI.
"""

import logging
from typing import Dict, List, Tuple, Iterable, Optional
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from whitespot.dataloader import DataLoader
from whitespot.record import Record
from whitespot.symbols import OUTPUTDIR

# Optional lib for ALS
try:
    import implicit  # type: ignore
    _HAS_IMPLICIT = True
except Exception:
    _HAS_IMPLICIT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ----------------------------
# Base interfaces
# ----------------------------


class BaseRecommender:
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def recommend_customer(self, customer_id: str, top_n: int = 10) -> List[Record]:
        raise NotImplementedError

    def recommend_all(self, top_n: int = 10) -> pd.DataFrame:
        rows: List[Dict] = []
        for cust in self.customers():
            for rank, r in enumerate(self.recommend_customer(cust, top_n=top_n), start=1):
                rows.append({
                    "CustomerID": r.customer,
                    "rank": rank,
                    "PartID": r.part,
                    "score": r.score,
                    "reason_from_part": r.reason_part,
                    "sim_to_reason": r.reason_score,
                })
        return pd.DataFrame(rows)

    def customers(self) -> Iterable[str]:
        raise NotImplementedError

# ----------------------------
# Cosine recommender
# ----------------------------

class CosineRecommender(BaseRecommender):
    def __init__(self, min_support: int = 3, use_qty: bool = False) -> None:
        self.min_support = min_support
        self.use_qty = use_qty
        self.ci: Optional[pd.DataFrame] = None
        self.sim: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "CosineRecommender":
        if self.use_qty:
            ci = (
                df.pivot_table(index="CustomerID", columns="PartID", values="QtyInBasket", aggfunc="sum", fill_value=0)
                .astype(np.float32)
            )
        else:
            ci = (
                df.assign(v=1)
                  .pivot_table(index="CustomerID", columns="PartID", values="v", aggfunc="max", fill_value=0)
                  .astype(np.float32)
            )
        support = ci.sum(axis=0)
        keep = support[support >= self.min_support].index
        ci = ci[keep]
        norm = np.sqrt((ci ** 2).sum(axis=0)).replace(0, 1.0)
        sim = pd.DataFrame(ci.T @ (ci / norm), index=ci.columns, columns=ci.columns)
        np.fill_diagonal(sim.values, 0.0)
        self.ci, self.sim = ci, sim
        return self

    def customers(self) -> Iterable[str]:
        return [] if self.ci is None else self.ci.index.tolist()

    def recommend_customer(self, customer_id: str, top_n: int = 10) -> List[Record]:
        if self.ci is None or self.sim is None or customer_id not in self.ci.index:
            return []
        owned = self.ci.loc[customer_id]
        owned_items = owned[owned > 0].index.intersection(self.sim.index)
        if owned_items.empty:
            return []
        sim_scores = self.sim[owned_items].sum(axis=1)
        sim_scores = sim_scores.drop(index=owned_items, errors="ignore")
        best_src = self.sim[owned_items].idxmax(axis=1)
        best_sim = self.sim[owned_items].max(axis=1)
        top = sim_scores.sort_values(ascending=False).head(top_n)
        return [
            Record(customer=customer_id, part=pid, score=float(top.loc[pid]),
                reason_part=str(best_src.loc[pid]), reason_score=float(best_sim.loc[pid]))
            for pid in top.index
        ]

# ----------------------------
# Co-Visitation recommender
# ----------------------------

class CoVisRecommender(BaseRecommender):
    def __init__(self, topk_neighbors: int = 100, score: str = "lift", min_support: int = 3) -> None:
        self.topk = topk_neighbors
        self.score = score  # "lift" | "count" | "jaccard"
        self.min_support = min_support
        self._neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self._owned_map: Optional[pd.DataFrame] = None

    def _pair_counts_and_support(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, int]:
        baskets = df[["OrderDocID", "PartID"]].drop_duplicates()
        total_baskets = baskets["OrderDocID"].nunique()
        supp = baskets.groupby("PartID")["OrderDocID"].nunique().rename("support")
        pairs = baskets.merge(baskets, on="OrderDocID")
        pairs = pairs[pairs["PartID_x"] < pairs["PartID_y"]]
        cooc = pairs.groupby(["PartID_x", "PartID_y"]).size().rename("cooc").reset_index()
        return cooc, supp, int(total_baskets)

    def fit(self, df: pd.DataFrame) -> "CoVisRecommender":
        cooc, supp, total_baskets = self._pair_counts_and_support(df)
        keep = set(supp[supp >= self.min_support].index)
        cooc = cooc[cooc["PartID_x"].isin(keep) & cooc["PartID_y"].isin(keep)]
        if self.score == "lift" or self.score == "jaccard":
            cooc = (
                cooc.merge(supp.rename("supp_x"), left_on="PartID_x", right_index=True)
                    .merge(supp.rename("supp_y"), left_on="PartID_y", right_index=True)
            )
        if self.score == "lift":
            cooc["score"] = cooc["cooc"] * total_baskets / (cooc["supp_x"] * cooc["supp_y"])
        elif self.score == "jaccard":
            cooc["score"] = cooc["cooc"] / (cooc["supp_x"] + cooc["supp_y"] - cooc["cooc"])
        else:
            cooc["score"] = cooc["cooc"].astype(float)
        # build neighbors
        nbrs: Dict[str, List[Tuple[str, float]]] = {}
        for (a, b, s) in cooc[["PartID_x", "PartID_y", "score"]].itertuples(index=False):
            for src, dst in ((a, b), (b, a)):
                lst = nbrs.setdefault(src, [])
                lst.append((dst, float(s)))
        for k in list(nbrs.keys()):
            lst = nbrs[k]
            lst.sort(key=lambda t: t[1], reverse=True)
            nbrs[k] = lst[:self.topk]
        self._neighbors = nbrs
        self._owned_map = (
            df.assign(v=1).pivot_table(index="CustomerID", columns="PartID", values="v", aggfunc="max", fill_value=0)
        )
        return self

    def customers(self) -> Iterable[str]:
        return [] if self._owned_map is None else self._owned_map.index.tolist()

    def recommend_customer(self, customer_id: str, top_n: int = 10) -> List[Record]:
        if self._owned_map is None or customer_id not in self._owned_map.index:
            return []
        owned = self._owned_map.loc[customer_id]
        owned_set = set(owned[owned > 0].index.tolist())
        scores: Dict[str, float] = {}
        best: Dict[str, Tuple[str, float]] = {}
        for src in owned_set:
            for dst, s in self._neighbors.get(src, []):
                if dst in owned_set:
                    continue
                scores[dst] = scores.get(dst, 0.0) + s
                if (dst not in best) or (s > best[dst][1]):
                    best[dst] = (src, s)
        ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top_n]
        return [Record(customer=customer_id, part=pid, score=float(sc), reason_part=best[pid][0], reason_score=float(best[pid][1])) for pid, sc in ranked]

# ----------------------------
# ALS recommender
# ----------------------------

class ALSRecommender(BaseRecommender):
    def __init__(self, factors: int = 64, iterations: int = 15, regularization: float = 0.01, alpha: float = 40.0, use_qty: bool = False) -> None:
        if not _HAS_IMPLICIT:
            raise RuntimeError("'implicit' not installed. Install via: pip install implicit")
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.use_qty = use_qty
        self.X: Optional[csr_matrix] = None
        self.cust_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}
        self.inv_cust: Dict[int, str] = {}
        self.inv_item: Dict[int, str] = {}
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization
        )

    def _build_sparse(self, df: pd.DataFrame) -> None:
        custs = df["CustomerID"].astype(str).unique()
        items = df["PartID"].astype(str).unique()
        self.cust_index = {c: i for i, c in enumerate(custs)}
        self.item_index = {p: i for i, p in enumerate(items)}
        rows = df["CustomerID"].map(self.cust_index).astype(int)
        cols = df["PartID"].map(self.item_index).astype(int)
        vals = (df["QtyInBasket"].astype(np.float32) if self.use_qty else pd.Series(1.0, index=df.index))
        X = coo_matrix((vals.values.astype(np.float32), (rows.values, cols.values)), shape=(len(custs), len(items))).tocsr()
        self.X = X
        self.inv_cust = {v: k for k, v in self.cust_index.items()}
        self.inv_item = {v: k for k, v in self.item_index.items()}

    def fit(self, df: pd.DataFrame) -> "ALSRecommender":
        self._build_sparse(df)
        assert self.X is not None
        C = (self.alpha * self.X).astype(np.float32)
        self.model.fit(C)
        return self

    def customers(self) -> Iterable[str]:
        return list(self.cust_index.keys())

    def recommend_customer(self, customer_id: str, top_n: int = 10) -> List[Record]:
        if self.X is None or customer_id not in self.cust_index:
            return []
        u = self.cust_index[customer_id]
        recs = self.model.recommend(userid=u, user_items=self.X[u], N=top_n, filter_already_liked_items=True)
        # Handle both return signatures: (ids, scores) or list[(id, score)]
        if isinstance(recs, tuple) and len(recs) == 2:
            ids, scores = recs
            pairs = list(zip(ids, scores))
        else:
            pairs = list(recs)
        out: List[Record] = []
        for i_idx, score in pairs:
            part = self.inv_item[int(i_idx)]
            out.append(Record(customer=customer_id, part=part, score=float(score), reason_part="ALS", reason_score=None))
        return out

# ----------------------------
# Evaluation
# ----------------------------

class Evaluator:
    @staticmethod
    def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        order_dates = df.groupby(["CustomerID", "OrderDocID"])['OrderDocDate'].max().rename('OrderDate').reset_index()
        last_orders = order_dates.sort_values(["CustomerID", "OrderDate"]).groupby("CustomerID").tail(1)
        test_keys = set(zip(last_orders["CustomerID"], last_orders["OrderDocID"]))
        is_test = df.apply(lambda r: (r["CustomerID"], r["OrderDocID"]) in test_keys, axis=1)
        test = df[is_test].copy()
        train = df[~is_test].copy()
        return train, test

    @staticmethod
    def hitrate_at_k(model: BaseRecommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> Tuple[float, float]:
        if test.empty:
            return 0.0, 0.0
        customers = test["CustomerID"].unique()
        hits, ready = 0, 0
        truth = test.groupby("CustomerID")["PartID"].apply(lambda s: set(s.astype(str))).to_dict()
        for cust in customers:
            recs = model.recommend_customer(cust, top_n=k)
            if not recs:
                continue
            ready += 1
            pred = {r.part for r in recs}
            if len(pred & truth.get(cust, set())) > 0:
                hits += 1
        return hits / max(1, len(customers)), ready / max(1, len(customers))

# ----------------------------
# Orchestrator & CLI
# ----------------------------

class Orchestrator:
    def __init__(self) -> None:
        self.loader = DataLoader()

    def run(self,
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
        stats = df.groupby("PartID").agg(customers=("CustomerID","nunique"), baskets=("OrderDocID","nunique"), last_date=("OrderDocDate","max"), total_qty=("QtyInBasket","sum")).reset_index()
        stats.to_csv(OUTPUTDIR / "item_stats.csv", index=False, sep=";", encoding="utf-8")

        # Choose engine
        if engine == "baseline":
            model: BaseRecommender = CosineRecommender(min_support=min_support)
        elif engine == "covis":
            model = CoVisRecommender(topk_neighbors=covis_topk, score=covis_score, min_support=min_support)
        else:
            model = ALSRecommender(factors=als_factors, iterations=als_iterations, regularization=als_regularization, alpha=als_alpha, use_qty=als_use_qty)

        model.fit(df)

        # Export recommendations
        rec_df = model.recommend_all(top_n=top_n)
        part1 = DataLoader.part_dict(df)
        rec_df["PartDesc"] = rec_df["PartID"].map(lambda pid: part1.loc[pid, "PartDesc1"] if pid in part1.index else "")
        suffix = f"_{engine}"
        rec_df.to_csv(OUTPUTDIR / f"recommendations_per_customer{suffix}.csv", index=False, sep=";", encoding="utf-8")

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
        pd.Series(meta).to_csv(OUTPUTDIR / f"run_metadata{suffix}.csv", sep=";", header=False)
        logging.info("Saved outputs to %s", OUTPUTDIR)
        logging.info("[%s] HitRate@%d = %.3f | Coverage = %.3f", engine, top_n, hit, cov)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whitespot analysis (class-based)")
    p.add_argument("--nrows", type=int, default=5000)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--engine", choices=["baseline","covis","als"], default="baseline")
    p.add_argument("--covis-topk", type=int, default=100)
    p.add_argument("--covis-score", choices=["lift","count","jaccard"], default="lift")
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
