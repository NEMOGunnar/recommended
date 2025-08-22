# ----------------------------
# Cosine recommender
# ----------------------------

from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
from whitespot.recommender.baserecommender import BaseRecommender
from whitespot.record import Record


class CosineRecommender(BaseRecommender):
    def __init__(self, min_support: int = 3, use_qty: bool = False) -> None:
        self.min_support = min_support
        self.use_qty = use_qty
        self.ci: Optional[pd.DataFrame] = None
        self.sim: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "CosineRecommender":
        if self.use_qty:
            ci = df.pivot_table(
                index="CustomerID",
                columns="PartID",
                values="QtyInBasket",
                aggfunc="sum",
                fill_value=0,
            ).astype(np.float32)
        else:
            ci = (
                df.assign(v=1)
                .pivot_table(
                    index="CustomerID",
                    columns="PartID",
                    values="v",
                    aggfunc="max",
                    fill_value=0,
                )
                .astype(np.float32)
            )
        support = ci.sum(axis=0)
        keep = support[support >= self.min_support].index
        ci = ci[keep]
        norm = np.sqrt((ci**2).sum(axis=0)).replace(0, 1.0)
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
            Record(
                customer=customer_id,
                part=pid,
                score=float(top.loc[pid]),
                reason_part=str(best_src.loc[pid]),
                reason_score=float(best_sim.loc[pid]),
            )
            for pid in top.index
        ]
