# ----------------------------
# Evaluation
# ----------------------------

from typing import Tuple

import pandas as pd
from whitespot.recommender.baserecommender import BaseRecommender
    
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


