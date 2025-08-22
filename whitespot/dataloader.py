# ----------------------------
# Data layer
# ----------------------------

import logging
from pathlib import Path
import pandas as pd

from whitespot.symbols import DATADIR, FILENAME


class DataLoader:
    """Loads and prepares the transactional data."""

    def __init__(self, datadir: Path = DATADIR, filename: str = FILENAME) -> None:
        self.datadir = datadir
        self.filename = filename

    def read(self, nrows: int = 5000) -> pd.DataFrame:
        """Read CSV.GZ, normalize types, aggregate duplicate lines per basket."""
        file_path = self.datadir / self.filename
        df = (
            pd.read_csv(
                file_path,
                compression="gzip",
                sep=";",
                encoding="utf-8",
                nrows=nrows,
                low_memory=False,
                usecols=[
                    "PartDesc1","PartDesc2","PartDesc3","PartDesc4",
                    "PartID","OrderDocLineQty","OrderDocID","OrderDocLineID",
                    "CustomerID","OrderDocDate",
                ],
            )
            .dropna(subset=["PartID", "OrderDocLineID", "CustomerID"])  # hygiene
            .copy()
        )
        df["PartID"] = df["PartID"].astype(str)
        df["CustomerID"] = df["CustomerID"].astype(str)
        df["OrderDocID"] = df["OrderDocID"].astype(str)
        df["OrderDocDate"] = pd.to_datetime(df["OrderDocDate"], errors="coerce")
        df = df.dropna(subset=["OrderDocDate"]).copy()

        keys = [
            "PartDesc1","PartDesc2","PartDesc3","PartDesc4",
            "PartID","OrderDocID","CustomerID","OrderDocDate",
        ]
        df = (
            df.groupby(keys, as_index=False)["OrderDocLineQty"].sum()
              .rename(columns={"OrderDocLineQty": "QtyInBasket"})
        )
        logging.info("Data read from %s, shape after cleaning: %s", file_path, df.shape)
        return df

    @staticmethod
    def part_dict(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values("OrderDocDate")
              .drop_duplicates("PartID")
              .set_index("PartID")["PartDesc1"].to_frame()
        )
