from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd
from whitespot.recommender.baserecommender import BaseRecommender
from whitespot.record import Record

# ----------------------------
# Co-Visitation recommender
# ----------------------------


class CoVisRecommender(BaseRecommender):
    def __init__(
        self, topk_neighbors: int = 100, score: str = "lift", min_support: int = 3
    ) -> None:
        self.topk = topk_neighbors
        self.score = score  # "lift" | "count" | "jaccard"
        self.min_support = min_support
        self._neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self._owned_map: Optional[pd.DataFrame] = None

    def _pair_counts_and_support(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, int]:
        baskets = df[["OrderDocID", "PartID"]].drop_duplicates()
        total_baskets = baskets["OrderDocID"].nunique()
        supp = baskets.groupby("PartID")["OrderDocID"].nunique().rename("support")
        pairs = baskets.merge(baskets, on="OrderDocID")
        pairs = pairs[pairs["PartID_x"] < pairs["PartID_y"]]
        cooc = (
            pairs.groupby(["PartID_x", "PartID_y"]).size().rename("cooc").reset_index()
        )
        return cooc, supp, int(total_baskets)

    def fit(self, df: pd.DataFrame) -> "CoVisRecommender":
        cooc, supp, total_baskets = self._pair_counts_and_support(df)
        keep = set(supp[supp >= self.min_support].index)
        cooc = cooc[cooc["PartID_x"].isin(keep) & cooc["PartID_y"].isin(keep)]
        if self.score == "lift" or self.score == "jaccard":
            cooc = cooc.merge(
                supp.rename("supp_x"), left_on="PartID_x", right_index=True
            ).merge(supp.rename("supp_y"), left_on="PartID_y", right_index=True)
        if self.score == "lift":
            cooc["score"] = (
                cooc["cooc"] * total_baskets / (cooc["supp_x"] * cooc["supp_y"])
            )
        elif self.score == "jaccard":
            cooc["score"] = cooc["cooc"] / (
                cooc["supp_x"] + cooc["supp_y"] - cooc["cooc"]
            )
        else:
            cooc["score"] = cooc["cooc"].astype(float)
        # build neighbors
        nbrs: Dict[str, List[Tuple[str, float]]] = {}
        for a, b, s in cooc[["PartID_x", "PartID_y", "score"]].itertuples(index=False):
            for src, dst in ((a, b), (b, a)):
                lst = nbrs.setdefault(src, [])
                lst.append((dst, float(s)))
        for k in list(nbrs.keys()):
            lst = nbrs[k]
            lst.sort(key=lambda t: t[1], reverse=True)
            nbrs[k] = lst[: self.topk]
        self._neighbors = nbrs
        self._owned_map = df.assign(v=1).pivot_table(
            index="CustomerID",
            columns="PartID",
            values="v",
            aggfunc="max",
            fill_value=0,
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
        return [
            Record(
                customer=customer_id,
                part=pid,
                score=float(sc),
                reason_part=best[pid][0],
                reason_score=float(best[pid][1]),
            )
            for pid, sc in ranked
        ]
