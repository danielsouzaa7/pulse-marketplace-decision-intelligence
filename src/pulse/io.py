import pandas as pd
from pulse.config import BRONZE, SILVER, GOLD

def _p(base, name):
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.parquet"

def write_bronze(df: pd.DataFrame, name: str) -> None: df.to_parquet(_p(BRONZE, name), index=False)
def read_bronze(name: str) -> pd.DataFrame: return pd.read_parquet(_p(BRONZE, name))
def write_silver(df: pd.DataFrame, name: str) -> None: df.to_parquet(_p(SILVER, name), index=False)
def read_silver(name: str) -> pd.DataFrame: return pd.read_parquet(_p(SILVER, name))
def write_gold(df: pd.DataFrame, name: str) -> None: df.to_parquet(_p(GOLD, name), index=False)
def read_gold(name: str) -> pd.DataFrame: return pd.read_parquet(_p(GOLD, name))
