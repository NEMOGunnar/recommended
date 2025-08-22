# ----------------------------
# Paths & constants
# ----------------------------
from pathlib import Path


DATADIR = Path(__file__).parent.parent / "data"
FILENAME = "nemo-00000000-0000-0000-0000-000000000001-Deliver-utf-8.foc.gz"
OUTPUTDIR = DATADIR / "outputs"
OUTPUTDIR.mkdir(parents=True, exist_ok=True)
