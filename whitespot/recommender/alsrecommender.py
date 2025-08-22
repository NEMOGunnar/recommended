# ----------------------------
# ALS recommender
# ----------------------------

from typing import Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from whitespot.recommender.baserecommender import BaseRecommender
from whitespot.record import Record
import implicit  

class ALSRecommender(BaseRecommender):
    def __init__(
        self,
        factors: int = 64,
        iterations: int = 15,
        regularization: float = 0.01,
        alpha: float = 40.0,
        use_qty: bool = False,
    ) -> None:
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
        vals = (
            df["QtyInBasket"].astype(np.float32)
            if self.use_qty
            else pd.Series(1.0, index=df.index)
        )
        X = coo_matrix(
            (vals.values.astype(np.float32), (rows.values, cols.values)),
            shape=(len(custs), len(items)),
        ).tocsr()
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
        recs = self.model.recommend(
            userid=u, user_items=self.X[u], N=top_n, filter_already_liked_items=True
        )
        # Handle both return signatures: (ids, scores) or list[(id, score)]
        if isinstance(recs, tuple) and len(recs) == 2:
            ids, scores = recs
            pairs = list(zip(ids, scores))
        else:
            pairs = list(recs)
        out: List[Record] = []
        for i_idx, score in pairs:
            part = self.inv_item[int(i_idx)]
            out.append(
                Record(
                    customer=customer_id,
                    part=part,
                    score=float(score),
                    reason_part="ALS",
                    reason_score=None,
                )
            )
        return out
