from datetime import date, timedelta
from pathlib import Path

SEED = 42
START_DATE = date(2026, 3, 15)
N_DAYS = 180
END_DATE = START_DATE + timedelta(days=N_DAYS - 1)   # 2026-09-10
AS_OF = END_DATE

N_CUSTOMERS = 25_000
N_MERCHANTS = 150
N_ZONES = 8
TARGET_ORDERS = 100_000
TARGET_CONVERSION = 0.179          # -> ~560k sessions
OPERATING_HOURS = range(10, 24)    # 14-hour window for availability snapshots
PEAK_HOURS = (11, 12, 13, 18, 19, 20)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE, SILVER, GOLD = DATA / "bronze", DATA / "silver", DATA / "gold"
GROUND_TRUTH = DATA / "ground_truth"
ARTIFACTS = ROOT / "artifacts"
SQL = ROOT / "sql"
