from typing import Dict, Iterable, List
import pandas as pd

from whitespot.record import Record


class BaseRecommender:
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def recommend_customer(self, customer_id: str, top_n: int = 10) -> List[Record]:
        raise NotImplementedError

    def recommend_all(self, top_n: int = 10) -> pd.DataFrame:
        rows: List[Dict] = []
        for cust in self.customers():
            for rank, r in enumerate(
                self.recommend_customer(cust, top_n=top_n), start=1
            ):
                rows.append(
                    {
                        "CustomerID": r.customer,
                        "rank": rank,
                        "PartID": r.part,
                        "score": r.score,
                        "reason_from_part": r.reason_part,
                        "sim_to_reason": r.reason_score,
                    }
                )
        return pd.DataFrame(rows)

    def customers(self) -> Iterable[str]:
        raise NotImplementedError
