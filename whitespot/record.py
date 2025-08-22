from dataclasses import dataclass
from typing import Optional


@dataclass
class Record:
    customer: str
    part: str
    score: float
    reason_part: Optional[str] = None
    reason_score: Optional[float] = None

