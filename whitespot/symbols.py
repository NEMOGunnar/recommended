# ----------------------------
# Paths & constants
# ----------------------------
from pathlib import Path


DATADIR = Path(__file__).parent.parent / "data"
FILENAME = "NemoDataExport.foc.gz"
OUTPUTDIR = DATADIR / "outputs"
OUTPUTDIR.mkdir(parents=True, exist_ok=True)
