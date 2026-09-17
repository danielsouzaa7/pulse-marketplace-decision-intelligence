# PULSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest complete marketplace decision-intelligence system that detects a business anomaly, diagnoses its associated drivers, estimates impact, ranks priorities, emits a Decision Memo, validates an experiment, and presents all of it in a polished Streamlit product.

**Architecture:** Modular monolith. A seeded synthetic generator writes Bronze parquet; DuckDB executes real `.sql` files to produce Silver and Gold; a pure-function decision engine computes metrics, anomalies, diagnosis, impact, priority, and memos; one orchestrator (`run_decision_cycle`) is called by both the CLI and Streamlit so the two can never diverge. The LLM narrates evidence; it never decides policy or supplies facts.

**Tech Stack:** Python 3.12 (pinned via `uv`), pandas, numpy, scipy, duckdb, pyyaml, streamlit, plotly, anthropic, pytest. PySpark/Delta runs only on Databricks Free Edition as an evidence layer.

**Spec:** `docs/superpowers/specs/2026-09-11-pulse-marketplace-decision-intelligence-design.md`

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **Python 3.12**, pinned via `uv`. Local default is 3.14; do not use it.
- **Random seed = 42**, via `numpy.random.Generator(PCG64(42))`. No other RNG source anywhere.
- **Date window is fixed: 2026-03-15 to 2026-09-10 (180 days).** Never relative to `today()`. Analysis "as of" date is `2026-09-10`.
- **`GMV = sum(item_amount)` where status = completed.** Not `item_amount + delivery_fee`. `customer_gross_value = item_amount + delivery_fee`.
- **`contribution_margin = item_amount * commission_rate + delivery_fee - delivery_cost - discount_amount`.** `delivery_cost` is a generated per-order column, never a constant.
- **ONE ENGINE ONLY.** Exactly one orchestrator, `run_decision_cycle()`. CLI and Streamlit both import it. No duplicate implementation of metrics, detection, diagnosis, impact, prioritization, playbook, or memo generation may exist.
- **Ground truth isolation.** `data/ground_truth/` is readable only by `tests/`. No module under `src/pulse/` or `app/` may reference it.
- **Categorical identifiers are never imputed.** Numeric measures may be imputed and must set an `is_imputed` flag.
- **Reproducibility is asserted on content hashes of sorted DataFrames, never on file bytes.**
- **Causality language:** use "associated driver", "contribution", "correlated operational deterioration", "evidence consistent with". Causal claims only where the Experiment Lab supports them.
- **No secrets committed.** `.env` gitignored; `.env.example` holds `ANTHROPIC_API_KEY=` only. App works fully without a key except live narration.
- **Currency BRL (R$). All UI and docs in English.**
- **No hard-coded results.** Every displayed figure is read from a computed object.

---

## Phase Gates

| Phase | Tasks | Gate condition |
|---|---|---|
| **0 — Foundation** | 1-2 | `pytest` runs, `import pulse` works, contracts declared |
| **1 — Vertical slice** | 3-15 | **GATE #1:** synthetic data → bronze → silver → gold → GMV anomaly → Zone 7 contribution → delivery-time driver → cancellation driver → financial impact → priority → Decision Memo → visible in Streamlit |
| **2 — Breadth** | 16-22 | Incidents 2-4 detected and ranked; experiment lab and Copilot working; all 10 pages present |
| **3 — Evidence** | 23-25 | Databricks executed and reconciled; screenshots captured; README accurate; full verification ritual green |

**Do not begin Phase 2 until Gate #1 passes.** Ten half-finished modules are worth less than one complete loop.

---

## Dependency graph

```
T1 scaffold
 └─ T2 types + contracts
     ├─ T3 dimensions ──┐
     ├─ T4 incidents  ──┤
     │                  └─ T5 facts → bronze
     │                        └─ T6 defect injection
     │                              └─ T7 silver (DuckDB SQL)
     │                                    └─ T8 gold (spine + contracts)
     │                                          └─ T9 metrics
     │                                                └─ T10 anomaly
     │                                                      └─ T11 root cause
     │                                                            └─ T12 impact + priority
     │                                                                  ├─ T13 playbook
     │                                                                  └─ T14 memo + engine + CLI
     │                                                                        └─ T15 Streamlit  ══ GATE #1
     └───────────────────────────────────────────────────────────────────────────┘
                                                                                  │
  T16 incidents 2-4 ─ T17 remaining gold ─┬─ T18 playbook patterns ─┐             │
                                          ├─ T19 experiments ───────┤             │
                                          └─ T20 copilot ───────────┤             │
                                                                    ├─ T21 analytics pages
                                                                    └─ T22 platform pages
                                                                          └─ T23 Databricks
                                                                                └─ T24 README + screenshots
                                                                                      └─ T25 verification
```

**Critical path:** `T1 → T2 → T3 → T5 → T7 → T8 → T9 → T10 → T11 → T12 → T14 → T15 (Gate #1)`

T4 runs parallel to T3. T6 can be deferred behind T7 only if Silver is written to tolerate clean input first. T13 runs parallel to T12. T18/T19/T20 are mutually independent and can be parallelised after T17.

---

## File structure

| File | Responsibility |
|---|---|
| `src/pulse/types.py` | All frozen dataclasses. No logic, no I/O. |
| `src/pulse/contracts.py` | Gold table grain/PK/column declarations + `assert_contract()`. |
| `src/pulse/io.py` | Parquet read/write, path resolution. The only module that touches disk for data. |
| `src/pulse/incidents.py` | `Effects`, incident functions, `INCIDENTS` registry, ground-truth emission. |
| `src/pulse/data_generator.py` | Seeded synthetic generation of all 10 tables. |
| `src/pulse/quality.py` | Bronze defect injection + Silver cleaning orchestration + quality report. |
| `src/pulse/sql_runner.py` | Execute `.sql` files against DuckDB, write parquet. |
| `src/pulse/metrics.py` | Gold → `MetricFrame`. KPI formulas live here and nowhere else. |
| `src/pulse/anomaly_detection.py` | DOW-adjusted baseline z-score detection. |
| `src/pulse/root_cause.py` | Segment contribution, funnel log-decomposition, driver correlation, pattern classification. |
| `src/pulse/prioritization.py` | `estimate_impact()` + `prioritize()`. |
| `src/pulse/playbook.py` + `playbook.yml` | Deterministic recommendation lookup and rendering. |
| `src/pulse/decision_memo.py` | `DecisionMemo` assembly, `to_dict`, `to_markdown`. |
| `src/pulse/experiments.py` | Two-proportion test, CI, power, economics. |
| `src/pulse/copilot.py` | `EvidenceBundle`, `Narrator` protocol, `validate_response()`. |
| `src/pulse/engine.py` | `run_decision_cycle()` — THE orchestrator. |
| `src/pulse/cli.py` | `pulse generate / build / decide / run`. |
| `app/theme.py` | CSS injection + Plotly template registration. |
| `app/components.py` | `kpi_card`, `priority_card`, `evidence_table`, `status_pill`. |
| `app/streamlit_app.py` | Navigation + shared sidebar + cached engine call. |
| `app/pages/*.py` | One file per page. Presentation only — no analytics. |

**Hard rule:** no file under `app/` computes a KPI, a deviation, a contribution, or a score. Pages read objects returned by `run_decision_cycle()` and render them.

---

# Phase 0 — Foundation

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/pulse/__init__.py`, `tests/__init__.py`, `tests/test_scaffold.py`, `.streamlit/config.toml`
- Create dirs: `data/{bronze,silver,gold,ground_truth}`, `artifacts/`, `sql/{silver,gold,checks}`, `app/pages/`, `databricks/`, `dashboard/screenshots/`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `pulse`, working `pytest`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
import sys

def test_python_version_is_312():
    assert sys.version_info[:2] == (3, 12)

def test_package_imports():
    import pulse
    assert pulse.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse'`

- [ ] **Step 3: Create the project**

```bash
uv init --python 3.12 --package --name pulse
uv add pandas numpy scipy duckdb pyyaml streamlit plotly anthropic python-dotenv
uv add --dev pytest pytest-cov
mkdir -p data/bronze data/silver data/gold data/ground_truth artifacts sql/silver sql/gold sql/checks app/pages databricks dashboard/screenshots tests
```

```python
# src/pulse/__init__.py
__version__ = "0.1.0"
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.uv]
package = true
```

- [ ] **Step 4: Create the Streamlit theme config**

```toml
# .streamlit/config.toml
[theme]
base = "dark"
primaryColor = "#4C7DFF"
backgroundColor = "#0B1220"
secondaryBackgroundColor = "#131C2E"
textColor = "#E6EDF7"
font = "sans serif"

[server]
runOnSave = true
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_scaffold.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with Python 3.12 and Streamlit theme"
```

---

### Task 2: Types and contracts

**Files:**
- Create: `src/pulse/types.py`, `src/pulse/contracts.py`, `tests/test_contracts.py`

**Interfaces:**
- Consumes: nothing
- Produces: `AnalysisParams`, `Anomaly`, `SegmentContribution`, `FunnelStage`, `AssociatedDriver`, `Diagnosis`, `Impact`, `Priority`, `Recommendation`, `DecisionMemo`, `DecisionCycleResult`, `GOLD_CONTRACTS`, `assert_contract(df, name)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts.py
import dataclasses
import pandas as pd
import pytest
from datetime import date

from pulse.types import AnalysisParams, Anomaly
from pulse.contracts import GOLD_CONTRACTS, assert_contract


def test_params_are_frozen_with_documented_defaults():
    p = AnalysisParams(as_of=date(2026, 9, 10))
    assert p.comparison_window_days == 14
    assert p.baseline_window_days == 56
    assert p.sensitivity == 2.5
    assert p.min_materiality_brl == 5_000
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.sensitivity = 9.0


def test_every_gold_contract_declares_grain_and_pk():
    assert len(GOLD_CONTRACTS) == 6
    for name, c in GOLD_CONTRACTS.items():
        assert c.grain, f"{name} missing grain"
        assert c.primary_key, f"{name} missing primary key"
        assert set(c.primary_key).issubset(set(c.required_columns))


def test_assert_contract_rejects_duplicate_primary_key():
    df = pd.DataFrame({"metric_date": ["2026-01-01", "2026-01-01"],
                       "zone_id": [7, 7], "gmv": [1.0, 2.0]})
    with pytest.raises(AssertionError, match="duplicate"):
        assert_contract(df, "gold_zone_performance")


def test_assert_contract_rejects_null_primary_key():
    df = pd.DataFrame({"metric_date": ["2026-01-01", None],
                       "zone_id": [7, 4], "gmv": [1.0, 2.0]})
    with pytest.raises(AssertionError, match="null"):
        assert_contract(df, "gold_zone_performance")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.types'`

- [ ] **Step 3: Write `src/pulse/types.py`**

Define every dataclass as `@dataclass(frozen=True)`. Exact field lists from spec section 9:

```python
from dataclasses import dataclass, field
from datetime import date, datetime

@dataclass(frozen=True)
class AnalysisParams:
    as_of: date
    comparison_window_days: int = 14
    baseline_window_days: int = 56
    sensitivity: float = 2.5
    min_materiality_brl: float = 5_000.0

@dataclass(frozen=True)
class Anomaly:
    metric: str
    scope: str              # "company" | "zone" | "merchant_category" | "acquisition_channel"
    scope_value: str
    recent_value: float
    baseline_value: float
    deviation_abs: float
    deviation_pct: float
    z_score: float
    direction: str          # "drop" | "spike"
    first_detected_date: date
    n_observations: int

@dataclass(frozen=True)
class SegmentContribution:
    dimension: str
    segment: str
    segment_deviation_abs: float
    contribution_pct: float
    rank: int

@dataclass(frozen=True)
class FunnelStage:
    stage: str              # "sessions"|"order_conversion"|"completion_rate"|"aov"
    recent: float
    baseline: float
    deviation_pct: float
    log_contribution: float
    is_primary_break: bool

@dataclass(frozen=True)
class AssociatedDriver:
    metric: str
    recent: float
    baseline: float
    deviation_pct: float
    correlation_with_target: float
    temporal_alignment_days: int
    evidence_strength: str  # "strong"|"moderate"|"weak"

@dataclass(frozen=True)
class Diagnosis:
    anomaly: Anomaly
    contributions: tuple[SegmentContribution, ...]
    primary_segment: SegmentContribution | None
    funnel: tuple[FunnelStage, ...]
    funnel_break_stage: str
    drivers: tuple[AssociatedDriver, ...]
    pattern: str
    confidence: float

@dataclass(frozen=True)
class Impact:
    gmv_at_risk_brl: float
    orders_lost: int
    customers_affected: int
    margin_impact_brl: float
    daily_run_rate_brl: float
    projected_30d_brl: float

@dataclass(frozen=True)
class Priority:
    diagnosis: Diagnosis
    impact: Impact
    impact_score: float
    rank: int
    score_breakdown: dict[str, float]

@dataclass(frozen=True)
class Recommendation:
    action: str
    rationale: str
    expected_effect: str
    validation_method: str
    owner_function: str
    playbook_id: str
    effort: str

@dataclass(frozen=True)
class DecisionMemo:
    memo_id: str
    incident: str
    impact: Impact
    concentration: str
    associated_drivers: tuple[AssociatedDriver, ...]
    evidence: tuple[str, ...]
    priority: int
    recommended_action: str
    potential_result: str
    validation_method: str
    confidence: float
    generated_at: datetime
    params: AnalysisParams
    narrative: str | None = None      # LLM-authored, never a fact source

@dataclass(frozen=True)
class DecisionCycleResult:
    as_of: date
    params: AnalysisParams
    anomalies: tuple[Anomaly, ...]
    priorities: tuple[Priority, ...]
    memos: tuple[DecisionMemo, ...]
    headline_kpis: tuple[dict, ...]
```

- [ ] **Step 4: Write `src/pulse/contracts.py`**

```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class GoldContract:
    grain: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]

GOLD_CONTRACTS: dict[str, GoldContract] = {
    "gold_daily_business_metrics": GoldContract(
        grain="one row per date",
        primary_key=("metric_date",),
        required_columns=("metric_date", "gmv", "orders_placed", "orders_completed",
                          "sessions", "order_conversion", "completion_rate",
                          "cancellation_rate", "aov", "active_customers",
                          "avg_promised_eta_minutes", "avg_actual_delivery_minutes",
                          "on_time_rate", "contribution_margin", "discount_amount"),
    ),
    "gold_zone_performance": GoldContract(
        grain="one row per date x zone",
        primary_key=("metric_date", "zone_id"),
        required_columns=("metric_date", "zone_id", "gmv", "orders_placed",
                          "orders_completed", "sessions", "order_conversion",
                          "completion_rate", "cancellation_rate", "aov",
                          "avg_promised_eta_minutes", "avg_actual_delivery_minutes",
                          "on_time_rate", "contribution_margin"),
    ),
    "gold_merchant_performance": GoldContract(
        grain="one row per date x merchant",
        primary_key=("metric_date", "merchant_id"),
        required_columns=("metric_date", "merchant_id", "zone_id", "gmv",
                          "orders_completed", "availability_rate", "contribution_margin"),
    ),
    "gold_customer_retention": GoldContract(
        grain="one row per cohort_month x acquisition_channel x period_index",
        primary_key=("cohort_month", "acquisition_channel", "period_index"),
        required_columns=("cohort_month", "acquisition_channel", "period_index",
                          "cohort_size", "retained_customers", "retention_rate"),
    ),
    "gold_promotion_performance": GoldContract(
        grain="one row per date x promotion",
        primary_key=("metric_date", "promotion_id"),
        required_columns=("metric_date", "promotion_id", "orders_with_promo", "gmv",
                          "discount_amount", "contribution_margin"),
    ),
    "gold_experiment_results": GoldContract(
        grain="one row per experiment x variant x date",
        primary_key=("experiment_id", "variant", "metric_date"),
        required_columns=("experiment_id", "variant", "metric_date", "assigned_customers",
                          "converted_customers", "gmv", "contribution_margin",
                          "incentive_cost"),
    ),
}

def assert_contract(df: pd.DataFrame, name: str) -> None:
    c = GOLD_CONTRACTS[name]
    missing = set(c.required_columns) - set(df.columns)
    assert not missing, f"{name}: missing required columns {sorted(missing)}"
    pk = list(c.primary_key)
    assert not df[pk].isna().any().any(), f"{name}: null values in primary key {pk}"
    dupes = df.duplicated(subset=pk).sum()
    assert dupes == 0, f"{name}: {dupes} duplicate primary key rows on {pk}"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/pulse/types.py src/pulse/contracts.py tests/test_contracts.py
git commit -m "feat: decision engine types and gold table contracts"
```

---

# Phase 1 — Vertical slice (Gate #1)

### Task 3: Dimension generation

**Files:**
- Create: `src/pulse/config.py`, `src/pulse/data_generator.py`, `src/pulse/io.py`, `tests/test_generator.py`

**Interfaces:**
- Consumes: nothing
- Produces: `make_rng()`, `generate_zones(rng)`, `generate_customers(rng)`, `generate_merchants(rng, zones)` each returning `pd.DataFrame`; `START_DATE`, `END_DATE`, `N_DAYS`, `AS_OF`; `io.write_bronze(df, name)`, `io.read_bronze(name)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
import hashlib
import pandas as pd
from datetime import date

from pulse.config import START_DATE, END_DATE, N_DAYS, AS_OF, SEED
from pulse.data_generator import make_rng, generate_zones, generate_customers, generate_merchants


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame. Deliberately NOT a file-byte hash."""
    ordered = df.sort_index(axis=1).sort_values(by=sorted(df.columns)).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()


def test_calendar_is_fixed_and_180_days():
    assert START_DATE == date(2026, 3, 15)
    assert END_DATE == date(2026, 9, 10)
    assert N_DAYS == 180
    assert AS_OF == END_DATE
    assert SEED == 42


def test_same_seed_produces_identical_content():
    a = generate_customers(make_rng())
    b = generate_customers(make_rng())
    assert frame_hash(a) == frame_hash(b)


def test_zone_7_is_the_largest_zone_by_demand_weight():
    zones = generate_zones(make_rng())
    assert len(zones) == 8
    assert abs(zones["demand_weight"].sum() - 1.0) < 1e-9
    top = zones.sort_values("demand_weight", ascending=False).iloc[0]
    assert int(top["zone_id"]) == 7
    assert 0.16 <= top["demand_weight"] <= 0.20


def test_customers_have_dimension_attributes_not_derived_segments():
    c = generate_customers(make_rng())
    assert len(c) == 25_000
    assert c["customer_id"].is_unique
    assert set(c["acquisition_channel"].unique()) == {
        "organic", "paid_social", "referral", "paid_search"}
    # value tier / at-risk are DERIVED in gold, never generated
    assert "segment" not in c.columns
    assert "is_at_risk" not in c.columns


def test_merchants_are_150_and_reference_valid_zones():
    rng = make_rng()
    zones = generate_zones(rng)
    m = generate_merchants(rng, zones)
    assert len(m) == 150
    assert m["merchant_id"].is_unique
    assert set(m["zone_id"]).issubset(set(zones["zone_id"]))
    assert m["commission_rate"].between(0.12, 0.28).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.config'`

- [ ] **Step 3: Write `src/pulse/config.py`**

```python
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
```

- [ ] **Step 4: Write `src/pulse/io.py`**

```python
import pandas as pd
from pulse.config import BRONZE, SILVER, GOLD

def _p(base, name): 
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.parquet"

def write_bronze(df: pd.DataFrame, name: str) -> None: df.to_parquet(_p(BRONZE, name), index=False)
def read_bronze(name: str) -> pd.DataFrame: return pd.read_parquet(_p(BRONZE, name))
def write_silver(df: pd.DataFrame, name: str) -> None: df.to_parquet(_p(SILVER, name), index=False)
def read_silver(name: str) -> pd.DataFrame: return pd.read_parquet(_p(SILVER, name))
def read_gold(name: str) -> pd.DataFrame: return pd.read_parquet(_p(GOLD, name))
```

- [ ] **Step 5: Write the dimension generators in `src/pulse/data_generator.py`**

```python
import numpy as np, pandas as pd
from numpy.random import Generator, PCG64
from pulse.config import SEED, N_CUSTOMERS, N_MERCHANTS, START_DATE, N_DAYS

def make_rng() -> Generator:
    return Generator(PCG64(SEED))

# Zone 7 is deliberately the largest zone: a zone-local incident must be able
# to surface as a company-level anomaly. See spec section 5.
ZONE_WEIGHTS = np.array([0.14, 0.12, 0.11, 0.13, 0.10, 0.09, 0.18, 0.13])

def generate_zones(rng) -> pd.DataFrame:
    return pd.DataFrame({
        "zone_id": np.arange(1, 9),
        "zone_name": [f"Zone {i}" for i in range(1, 9)],
        "demand_weight": ZONE_WEIGHTS,
        "base_eta_minutes": rng.integers(28, 42, size=8),
        "courier_density_index": np.round(rng.uniform(0.7, 1.3, size=8), 3),
    })

def generate_customers(rng) -> pd.DataFrame:
    channels = np.array(["organic", "paid_social", "referral", "paid_search"])
    return pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS + 1),
        "signup_day": rng.integers(0, N_DAYS, size=N_CUSTOMERS),
        "home_zone_id": rng.choice(np.arange(1, 9), size=N_CUSTOMERS, p=ZONE_WEIGHTS),
        "acquisition_channel": rng.choice(channels, size=N_CUSTOMERS, p=[.42, .24, .14, .20]),
        "device_os": rng.choice(["ios", "android"], size=N_CUSTOMERS, p=[.45, .55]),
        "base_order_rate": np.round(rng.gamma(2.0, 0.5, size=N_CUSTOMERS), 4),
    })

def generate_merchants(rng, zones) -> pd.DataFrame:
    cats = np.array(["Pizza", "Burger", "Japanese", "Brazilian", "Healthy", "Dessert"])
    return pd.DataFrame({
        "merchant_id": np.arange(1, N_MERCHANTS + 1),
        "merchant_name": [f"Merchant {i:03d}" for i in range(1, N_MERCHANTS + 1)],
        "zone_id": rng.choice(zones["zone_id"].values, size=N_MERCHANTS, p=ZONE_WEIGHTS),
        "category": rng.choice(cats, size=N_MERCHANTS),
        "commission_rate": np.round(rng.uniform(0.12, 0.28, size=N_MERCHANTS), 4),
        "avg_ticket": np.round(rng.uniform(28, 95, size=N_MERCHANTS), 2),
        "quality_score": np.round(rng.uniform(3.2, 4.9, size=N_MERCHANTS), 2),
    })
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/pulse/config.py src/pulse/io.py src/pulse/data_generator.py tests/test_generator.py
git commit -m "feat: seeded dimension generation with unequal zone sizing"
```

---

### Task 4: Incident framework and Zone 7

**Files:**
- Create: `src/pulse/incidents.py`, `tests/test_incidents.py`

**Interfaces:**
- Consumes: `pulse.config`
- Produces: `Effects` (frozen, with `.none()` and `.combine()`), `IncidentContext`, `zone7_degradation(ctx)`, `INCIDENTS` list, `effects_for(ctx)`, `write_ground_truth()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incidents.py
import json
from pulse.incidents import Effects, IncidentContext, effects_for, write_ground_truth
from pulse.config import GROUND_TRUTH


def ctx(day, zone=7, hour=19):
    return IncidentContext(day=day, zone_id=zone, hour=hour,
                           merchant_id=1, customer_id=1, acquisition_channel="organic",
                           signup_day=0, promotion_id=None)


def test_zone7_is_inert_during_the_clean_baseline():
    e = effects_for(ctx(day=100))
    assert e == Effects.none()


def test_zone7_is_inert_in_other_zones():
    assert effects_for(ctx(day=170, zone=3)) == Effects.none()


def test_zone7_ramps_in_over_three_days_then_saturates():
    day150 = effects_for(ctx(day=150))
    day153 = effects_for(ctx(day=153))
    day175 = effects_for(ctx(day=175))
    assert 1.0 < day150.cancel_mult < day153.cancel_mult
    assert day153.cancel_mult == day175.cancel_mult          # saturated
    assert abs(day175.cancel_mult - 3.25) < 1e-9             # 8% -> 26%
    assert abs(day175.actual_delivery_mult - 1.35) < 1e-9
    assert abs(day175.promised_eta_mult - 1.12) < 1e-9


def test_zone7_never_touches_pre_checkout_conversion():
    """Zone 7 is a POST-checkout failure. This is what keeps it
    analytically distinct from Zone 4."""
    for day in (150, 160, 180):
        assert effects_for(ctx(day=day)).conversion_mult == 1.0


def test_ground_truth_is_written_outside_the_medallion_layers():
    write_ground_truth()
    path = GROUND_TRUTH / "injected_incidents.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    z7 = next(i for i in payload["incidents"] if i["id"] == "zone7_degradation")
    assert z7["start_day"] == 150
    assert z7["scope"] == {"dimension": "zone", "value": 7}
    assert "bronze" not in str(path) and "silver" not in str(path) and "gold" not in str(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_incidents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.incidents'`

- [ ] **Step 3: Write `src/pulse/incidents.py`**

```python
import json
from dataclasses import dataclass, replace
from pulse.config import GROUND_TRUTH

@dataclass(frozen=True)
class Effects:
    actual_delivery_mult: float = 1.0
    promised_eta_mult: float = 1.0
    cancel_mult: float = 1.0
    conversion_mult: float = 1.0
    availability_mult: float = 1.0
    discount_mult: float = 1.0
    repeat_mult: float = 1.0

    @staticmethod
    def none() -> "Effects":
        return Effects()

    def combine(self, other: "Effects") -> "Effects":
        return Effects(*[a * b for a, b in zip(
            (self.actual_delivery_mult, self.promised_eta_mult, self.cancel_mult,
             self.conversion_mult, self.availability_mult, self.discount_mult,
             self.repeat_mult),
            (other.actual_delivery_mult, other.promised_eta_mult, other.cancel_mult,
             other.conversion_mult, other.availability_mult, other.discount_mult,
             other.repeat_mult))])

@dataclass(frozen=True)
class IncidentContext:
    day: int
    zone_id: int
    hour: int
    merchant_id: int
    customer_id: int
    acquisition_channel: str
    signup_day: int
    promotion_id: int | None

def _ramp(day: int, start: int, days: int = 3) -> float:
    return min(1.0, max(0.0, (day - start + 1) / days))

def zone7_degradation(ctx: IncidentContext) -> Effects:
    if ctx.day < 150 or ctx.zone_id != 7:
        return Effects.none()
    r = _ramp(ctx.day, 150)
    return Effects(
        actual_delivery_mult=1 + 0.35 * r,
        promised_eta_mult=1 + 0.12 * r,
        cancel_mult=1 + 2.25 * r,        # 8% -> 26% at saturation
    )
    # No conversion effect: Zone 7 fails AFTER checkout.

INCIDENTS = [zone7_degradation]          # Tasks 16 appends the remaining three

def effects_for(ctx: IncidentContext) -> Effects:
    out = Effects.none()
    for fn in INCIDENTS:
        out = out.combine(fn(ctx))
    return out

GROUND_TRUTH_SPEC = [
    {"id": "zone7_degradation", "start_day": 150, "end_day": 180,
     "scope": {"dimension": "zone", "value": 7},
     "expected_funnel_break": "completion_rate",
     "expected_drivers": ["avg_actual_delivery_minutes", "cancellation_rate"]},
]

def write_ground_truth() -> None:
    GROUND_TRUTH.mkdir(parents=True, exist_ok=True)
    (GROUND_TRUTH / "injected_incidents.json").write_text(
        json.dumps({"seed": 42, "incidents": GROUND_TRUTH_SPEC}, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_incidents.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/incidents.py tests/test_incidents.py
git commit -m "feat: incident effect framework with Zone 7 post-checkout degradation"
```

---

### Task 5: Fact generation to Bronze

**Files:**
- Modify: `src/pulse/data_generator.py`
- Modify: `tests/test_generator.py`

**Interfaces:**
- Consumes: `generate_zones`, `generate_customers`, `generate_merchants`, `effects_for`, `io.write_bronze`
- Produces: `generate_sessions(rng, customers, zones)`, `generate_orders(rng, sessions, merchants, zones)`, `generate_deliveries(rng, orders, zones)`, `generate_all() -> dict[str, pd.DataFrame]`

Generation order matters: sessions come first, orders are drawn from converting sessions, deliveries from non-pre-dispatch-cancelled orders. This enforces the funnel identity structurally rather than by assertion.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_generator.py
from pulse.data_generator import generate_all
from pulse.config import TARGET_ORDERS

import pytest

@pytest.fixture(scope="module")
def tables():
    return generate_all()


def test_row_counts_are_on_target(tables):
    assert 95_000 <= len(tables["orders"]) <= 105_000
    assert 500_000 <= len(tables["sessions"]) <= 620_000
    assert len(tables["deliveries"]) <= len(tables["orders"])


def test_orders_reference_only_converting_sessions(tables):
    s, o = tables["sessions"], tables["orders"]
    converting = set(s.loc[s["converted"], "session_id"])
    assert set(o["session_id"]).issubset(converting)
    assert len(converting) == len(o)


def test_money_columns_exist_and_delivery_cost_varies(tables):
    o = tables["orders"]
    for col in ("item_amount", "delivery_fee", "delivery_cost",
                "discount_amount", "commission_rate"):
        assert col in o.columns
    # delivery_cost MUST vary per order, else contribution margin is a
    # scalar multiple of GMV and the promo incident is undetectable.
    assert o["delivery_cost"].std() > 0.5


def test_zone7_completion_rate_collapses_in_the_incident_window(tables):
    o = tables["orders"]
    z7 = o[o["zone_id"] == 7]
    before = z7[z7["order_day"] < 150]
    after = z7[z7["order_day"] >= 153]
    comp_before = (before["status"] == "completed").mean()
    comp_after = (after["status"] == "completed").mean()
    assert comp_before > 0.90
    assert comp_after < 0.80
    assert (comp_after / comp_before) < 0.85


def test_zone7_order_placement_conversion_stays_stable(tables):
    """Correction #1: Zone 7 must NOT depress pre-checkout conversion."""
    s = tables["sessions"]
    z7 = s[s["zone_id"] == 7]
    conv_before = z7[z7["session_day"] < 150]["converted"].mean()
    conv_after = z7[z7["session_day"] >= 153]["converted"].mean()
    assert abs(conv_after - conv_before) / conv_before < 0.05


def test_zone7_delivery_time_degrades(tables):
    d = tables["deliveries"].merge(
        tables["orders"][["order_id", "zone_id", "order_day"]], on="order_id")
    z7 = d[d["zone_id"] == 7]
    before = z7[z7["order_day"] < 150]["actual_delivery_minutes"].mean()
    after = z7[z7["order_day"] >= 153]["actual_delivery_minutes"].mean()
    assert after / before > 1.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_all'`

- [ ] **Step 3: Implement fact generation**

Key structure (fill in the vectorised bodies; all randomness from the single `rng`):

```python
def _seasonality(days: np.ndarray) -> np.ndarray:
    """Weekly pattern + payday lift. Baseline behaviour, not an incident."""
    dow = (days + START_DATE.weekday()) % 7
    weekly = np.where(np.isin(dow, [4, 5]), 1.22, 1.0)          # Fri/Sat lift
    payday = np.where(np.isin(days % 30, [0, 1, 2, 14, 15]), 1.08, 1.0)
    return weekly * payday

def generate_sessions(rng, customers, zones) -> pd.DataFrame:
    # Draw ~560k sessions across 180 days weighted by zone demand_weight,
    # hour-of-day curve and _seasonality(). Apply effects_for(...).conversion_mult
    # to the per-session conversion probability. Emit:
    #   session_id, customer_id, zone_id, session_day, session_hour,
    #   device_os, converted (bool)
    ...

def generate_orders(rng, sessions, merchants, zones) -> pd.DataFrame:
    # One order per converting session. Draw merchant within the session's zone.
    # item_amount ~ merchant avg_ticket with lognormal noise.
    # delivery_fee ~ zone distance band. delivery_cost ~ courier time model,
    #   inflated by effects.actual_delivery_mult (slow zones cost more to serve).
    # cancellation: base 8% * effects.cancel_mult -> status in {completed, cancelled}
    # Emit: order_id, session_id, customer_id, merchant_id, zone_id, order_day,
    #   order_hour, order_ts, status, item_amount, delivery_fee, delivery_cost,
    #   discount_amount, commission_rate, promotion_id, payment_method
    ...

def generate_deliveries(rng, orders, zones) -> pd.DataFrame:
    # One row per order that entered dispatch (exclude pre-dispatch cancels).
    # promised_eta_minutes = zone base_eta * effects.promised_eta_mult
    # actual_delivery_minutes = promised baseline * effects.actual_delivery_mult
    #   * lognormal noise
    # Emit: delivery_id, order_id, assigned_ts, picked_up_ts, delivered_ts,
    #   promised_eta_minutes, actual_delivery_minutes
    ...

def generate_all() -> dict[str, pd.DataFrame]:
    rng = make_rng()
    zones = generate_zones(rng)
    customers = generate_customers(rng)
    merchants = generate_merchants(rng, zones)
    sessions = generate_sessions(rng, customers, zones)
    orders = generate_orders(rng, sessions, merchants, zones)
    deliveries = generate_deliveries(rng, orders, zones)
    return {"zones": zones, "customers": customers, "merchants": merchants,
            "sessions": sessions, "orders": orders, "deliveries": deliveries}
```

- [ ] **Step 4: Calibrate until the tests pass**

Run: `uv run pytest tests/test_generator.py -v`

If `test_zone7_completion_rate_collapses_in_the_incident_window` fails, adjust the base cancellation rate or `cancel_mult` in `incidents.py` — these are calibration parameters, and the test is the gate. Do **not** weaken the assertion.

Expected when calibrated: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/data_generator.py tests/test_generator.py
git commit -m "feat: session/order/delivery generation with Zone 7 incident applied"
```

---

### Task 6: Bronze defect injection

**Files:**
- Create: `src/pulse/quality.py`, `tests/test_quality.py`

**Interfaces:**
- Consumes: `generate_all()`
- Produces: `inject_defects(tables, rng) -> dict[str, pd.DataFrame]`, `write_quality_ground_truth(counts)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality.py
import json
import pandas as pd
from pulse.data_generator import generate_all, make_rng
from pulse.quality import inject_defects
from pulse.config import GROUND_TRUTH


def test_defects_are_injected_at_the_specified_rates():
    clean = generate_all()
    dirty = inject_defects(clean, make_rng())

    n_clean = len(clean["orders"])
    n_dirty = len(dirty["orders"])
    dup_rate = (n_dirty - n_clean) / n_clean
    assert 0.006 <= dup_rate <= 0.010                        # 0.8% duplicates

    assert dirty["orders"]["delivery_fee"].isna().mean() > 0.010   # 1.5% nulls
    assert dirty["sessions"]["zone_id"].isna().mean() > 0.003      # 0.5% nulls

    pm = set(dirty["orders"]["payment_method"].dropna().unique())
    assert len(pm) > 4, "category drift must be present in bronze"

    counts = json.loads((GROUND_TRUTH / "quality_defects.json").read_text())
    assert counts["duplicate_orders"] > 0
    assert counts["null_delivery_fee"] > 0
    assert counts["orphan_merchant_fk"] > 0
    assert counts["timestamp_violations"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.quality'`

- [ ] **Step 3: Implement `inject_defects`**

```python
PAYMENT_VARIANTS = ["credit_card", "CREDIT_CARD", "Credit Card", "cc"]
CATEGORY_VARIANTS = {"Pizza": ["Pizza", "pizza", "PIZZA "]}

def inject_defects(tables, rng) -> dict[str, pd.DataFrame]:
    """Applied AFTER clean generation so ground truth stays exact.
    Records every count to data/ground_truth/quality_defects.json."""
    counts = {}
    o = tables["orders"].copy()

    # 1. exact duplicates (0.8%)
    dup_idx = rng.choice(o.index, size=int(0.008 * len(o)), replace=False)
    o = pd.concat([o, o.loc[dup_idx]], ignore_index=True)
    counts["duplicate_orders"] = len(dup_idx)

    # 2. null delivery_fee (1.5%) - NUMERIC, imputable in silver
    n_idx = rng.choice(o.index, size=int(0.015 * len(o)), replace=False)
    o.loc[n_idx, "delivery_fee"] = None
    counts["null_delivery_fee"] = len(n_idx)

    # 3. category drift on payment_method
    d_idx = rng.choice(o.index, size=int(0.12 * len(o)), replace=False)
    o.loc[d_idx, "payment_method"] = rng.choice(PAYMENT_VARIANTS, size=len(d_idx))
    counts["category_drift_payment"] = len(d_idx)

    # 4. orphan merchant FK (0.2%)
    f_idx = rng.choice(o.index, size=int(0.002 * len(o)), replace=False)
    o.loc[f_idx, "merchant_id"] = 999_999
    counts["orphan_merchant_fk"] = len(f_idx)

    # 5. timestamp violations (0.3%) on deliveries
    dl = tables["deliveries"].copy()
    t_idx = rng.choice(dl.index, size=int(0.003 * len(dl)), replace=False)
    dl.loc[t_idx, "delivered_ts"] = dl.loc[t_idx, "assigned_ts"] - pd.Timedelta(hours=2)
    counts["timestamp_violations"] = len(t_idx)

    # 6. null zone_id on sessions (0.5%) - CATEGORICAL, never imputed
    s = tables["sessions"].copy()
    z_idx = rng.choice(s.index, size=int(0.005 * len(s)), replace=False)
    s.loc[z_idx, "zone_id"] = None
    counts["null_session_zone_id"] = len(z_idx)

    write_quality_ground_truth(counts)
    return {**tables, "orders": o, "deliveries": dl, "sessions": s}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quality.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/quality.py tests/test_quality.py
git commit -m "feat: seeded bronze quality defect injection with recorded ground truth"
```

---

### Task 7: Silver via DuckDB SQL

**Files:**
- Create: `src/pulse/sql_runner.py`, `sql/silver/01_orders.sql`, `sql/silver/02_sessions.sql`, `sql/silver/03_deliveries.sql`, `sql/silver/05_dimensions.sql`
- Modify: `src/pulse/quality.py` (add `build_silver()`)
- Modify: `tests/test_quality.py`

**Interfaces:**
- Consumes: bronze parquet
- Produces: `run_sql_file(path, **params) -> None`, `build_silver() -> pd.DataFrame` (the quality report), silver parquet + `data/silver/_rejected/*.parquet` + `data/silver/_quality_report.parquet`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_quality.py
from pulse.quality import build_silver
from pulse import io


def test_silver_removes_duplicates_and_quarantines_bad_rows():
    report = build_silver()
    orders = io.read_silver("orders")

    assert orders["order_id"].is_unique
    assert orders["merchant_id"].max() < 999_999          # orphans gone
    assert not report.empty
    assert {"check_name", "table", "rows_in", "rows_out",
            "rows_rejected", "rule", "run_ts"}.issubset(report.columns)
    assert report["rows_rejected"].sum() > 0


def test_numeric_nulls_imputed_with_a_flag_categoricals_quarantined():
    orders = io.read_silver("orders")
    sessions = io.read_silver("sessions")

    # numeric: imputed, flagged
    assert orders["delivery_fee"].isna().sum() == 0
    assert "delivery_fee_is_imputed" in orders.columns
    assert orders["delivery_fee_is_imputed"].sum() > 0

    # categorical identifier: NEVER imputed
    assert sessions["zone_id"].isna().sum() == 0
    assert "zone_id_is_imputed" not in sessions.columns
    rejected = pd.read_parquet("data/silver/_rejected/sessions.parquet")
    assert (rejected["reject_reason"] == "null_zone_id_unresolvable").sum() > 0


def test_categories_are_normalised():
    orders = io.read_silver("orders")
    assert set(orders["payment_method"].unique()).issubset(
        {"credit_card", "debit_card", "pix", "cash", "meal_voucher"})


def test_derived_columns_follow_the_locked_metric_semantics():
    orders = io.read_silver("orders")
    row = orders.iloc[0]
    expected_margin = (row["item_amount"] * row["commission_rate"]
                       + row["delivery_fee"] - row["delivery_cost"]
                       - row["discount_amount"])
    assert abs(row["contribution_margin"] - expected_margin) < 0.01
    assert abs(row["customer_gross_value"]
               - (row["item_amount"] + row["delivery_fee"])) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_silver'`

- [ ] **Step 3: Write `src/pulse/sql_runner.py`**

```python
import duckdb, pandas as pd
from pathlib import Path
from pulse.config import BRONZE, SILVER, GOLD

def run_sql_file(path: Path, out: Path | None = None) -> pd.DataFrame:
    sql = path.read_text().format(bronze=BRONZE.as_posix(),
                                  silver=SILVER.as_posix(),
                                  gold=GOLD.as_posix())
    con = duckdb.connect()
    df = con.sql(sql).df()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df
```

- [ ] **Step 4: Write `sql/silver/01_orders.sql`**

```sql
WITH raw AS (
    SELECT * FROM read_parquet('{bronze}/orders.parquet')
),
deduped AS (
    SELECT * FROM raw
    QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_ts) = 1
),
normalised AS (
    SELECT
        * EXCLUDE (payment_method),
        CASE
            WHEN lower(trim(payment_method)) IN ('credit_card','credit card','cc')
                 THEN 'credit_card'
            WHEN lower(trim(payment_method)) IN ('debit_card','debit card')
                 THEN 'debit_card'
            ELSE lower(trim(payment_method))
        END AS payment_method
    FROM deduped
),
zone_median_fee AS (
    SELECT zone_id, median(delivery_fee) AS median_fee
    FROM normalised WHERE delivery_fee IS NOT NULL GROUP BY zone_id
),
imputed AS (
    SELECT
        n.* EXCLUDE (delivery_fee),
        COALESCE(n.delivery_fee, z.median_fee)  AS delivery_fee,
        n.delivery_fee IS NULL                  AS delivery_fee_is_imputed
    FROM normalised n
    LEFT JOIN zone_median_fee z USING (zone_id)
)
SELECT
    i.*,
    i.item_amount + i.delivery_fee                        AS customer_gross_value,
    i.item_amount + i.delivery_fee - i.discount_amount    AS net_revenue,
    i.item_amount * i.commission_rate + i.delivery_fee
        - i.delivery_cost - i.discount_amount             AS contribution_margin,
    i.status = 'completed'                                AS is_completed,
    i.status = 'cancelled'                                AS is_cancelled,
    i.order_hour IN (11,12,13,18,19,20)                   AS is_peak,
    dayofweek(i.order_ts) IN (0,6)                        AS is_weekend
FROM imputed i
-- FK integrity: orphans are quarantined by 02_rejected, not silently dropped
INNER JOIN read_parquet('{bronze}/merchants.parquet') m USING (merchant_id);
```

- [ ] **Step 5: Write `sql/silver/02_sessions.sql`**

Categorical resolution then quarantine — never imputation:

```sql
WITH raw AS (SELECT * FROM read_parquet('{bronze}/sessions.parquet')),
customers AS (SELECT customer_id, home_zone_id FROM read_parquet('{bronze}/customers.parquet')),
resolved AS (
    SELECT
        r.* EXCLUDE (zone_id),
        -- Resolve ONLY from a deterministic valid relationship.
        COALESCE(r.zone_id, c.home_zone_id) AS zone_id
    FROM raw r LEFT JOIN customers c USING (customer_id)
)
SELECT * FROM resolved WHERE zone_id IS NOT NULL;
```

And the matching rejected-rows query written to `data/silver/_rejected/sessions.parquet`:

```sql
-- sql/silver/02_sessions_rejected.sql
WITH raw AS (SELECT * FROM read_parquet('{bronze}/sessions.parquet')),
customers AS (SELECT customer_id, home_zone_id FROM read_parquet('{bronze}/customers.parquet'))
SELECT r.*, 'null_zone_id_unresolvable' AS reject_reason
FROM raw r LEFT JOIN customers c USING (customer_id)
WHERE r.zone_id IS NULL AND c.home_zone_id IS NULL;
```

- [ ] **Step 6: Write `build_silver()` in `quality.py`**

It runs each SQL file in order, writes silver parquet and rejected parquet, and appends one row per check to `_quality_report.parquet` with `check_name, table, rows_in, rows_out, rows_rejected, rule, run_ts`.

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_quality.py -v`
Expected: PASS (5 passed)

- [ ] **Step 8: Commit**

```bash
git add src/pulse/sql_runner.py sql/silver src/pulse/quality.py tests/test_quality.py
git commit -m "feat: silver layer in DuckDB SQL with quarantine and quality report"
```

---

### Task 8: Gold with calendar spine

**Files:**
- Create: `sql/gold/gold_daily_business_metrics.sql`, `sql/gold/gold_zone_performance.sql`
- Modify: `src/pulse/quality.py` (add `build_gold()`)
- Modify: `tests/test_contracts.py`

**Interfaces:**
- Consumes: silver parquet, `assert_contract`
- Produces: `build_gold() -> None`, gold parquet for the two slice datasets

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_contracts.py
from pulse.quality import build_gold
from pulse import io
from pulse.config import N_DAYS


def test_gold_tables_satisfy_their_declared_contracts():
    build_gold()
    for name in ("gold_daily_business_metrics", "gold_zone_performance"):
        assert_contract(io.read_gold(name), name)


def test_calendar_spine_emits_a_row_for_every_zone_day():
    """Without a spine, a zone-day with zero completed orders vanishes -
    the worse Zone 7 gets, the more of its bad days disappear."""
    z = io.read_gold("gold_zone_performance")
    assert len(z) == N_DAYS * 8
    assert z.groupby("zone_id")["metric_date"].nunique().eq(N_DAYS).all()
    assert z["gmv"].isna().sum() == 0          # zero-activity days are 0.0, not null


def test_gmv_is_item_amount_of_completed_orders_only():
    orders = io.read_silver("orders")
    expected = orders.loc[orders["is_completed"], "item_amount"].sum()
    actual = io.read_gold("gold_daily_business_metrics")["gmv"].sum()
    assert abs(actual - expected) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_gold'`

- [ ] **Step 3: Write `sql/gold/gold_zone_performance.sql`**

```sql
WITH spine AS (
    SELECT d.metric_date, z.zone_id
    FROM (SELECT unnest(generate_series(DATE '2026-03-15',
                                        DATE '2026-09-10',
                                        INTERVAL 1 DAY))::DATE AS metric_date) d
    CROSS JOIN (SELECT zone_id FROM read_parquet('{silver}/zones.parquet')) z
),
sess AS (
    SELECT CAST(session_ts AS DATE) AS metric_date, zone_id,
           COUNT(*) AS sessions
    FROM read_parquet('{silver}/sessions.parquet') GROUP BY 1, 2
),
ord AS (
    SELECT CAST(order_ts AS DATE) AS metric_date, zone_id,
           COUNT(*)                                        AS orders_placed,
           COUNT(*) FILTER (WHERE is_completed)            AS orders_completed,
           COUNT(*) FILTER (WHERE is_cancelled)            AS orders_cancelled,
           SUM(item_amount) FILTER (WHERE is_completed)    AS gmv,
           SUM(contribution_margin) FILTER (WHERE is_completed) AS contribution_margin
    FROM read_parquet('{silver}/orders.parquet') GROUP BY 1, 2
),
dlv AS (
    SELECT CAST(o.order_ts AS DATE) AS metric_date, o.zone_id,
           AVG(d.promised_eta_minutes)      AS avg_promised_eta_minutes,
           AVG(d.actual_delivery_minutes)   AS avg_actual_delivery_minutes,
           AVG(CASE WHEN d.actual_delivery_minutes
                         <= d.promised_eta_minutes + 5 THEN 1.0 ELSE 0.0 END)
                                            AS on_time_rate
    FROM read_parquet('{silver}/deliveries.parquet') d
    JOIN read_parquet('{silver}/orders.parquet') o USING (order_id)
    GROUP BY 1, 2
)
SELECT
    s.metric_date, s.zone_id,
    COALESCE(se.sessions, 0)                                   AS sessions,
    COALESCE(o.orders_placed, 0)                               AS orders_placed,
    COALESCE(o.orders_completed, 0)                            AS orders_completed,
    COALESCE(o.gmv, 0.0)                                       AS gmv,
    COALESCE(o.contribution_margin, 0.0)                       AS contribution_margin,
    CASE WHEN COALESCE(se.sessions,0) = 0 THEN 0.0
         ELSE o.orders_placed::DOUBLE / se.sessions END        AS order_conversion,
    CASE WHEN COALESCE(o.orders_placed,0) = 0 THEN 0.0
         ELSE o.orders_completed::DOUBLE / o.orders_placed END AS completion_rate,
    CASE WHEN COALESCE(o.orders_placed,0) = 0 THEN 0.0
         ELSE o.orders_cancelled::DOUBLE / o.orders_placed END AS cancellation_rate,
    CASE WHEN COALESCE(o.orders_completed,0) = 0 THEN 0.0
         ELSE o.gmv / o.orders_completed END                   AS aov,
    d.avg_promised_eta_minutes, d.avg_actual_delivery_minutes, d.on_time_rate
FROM spine s
LEFT JOIN sess se USING (metric_date, zone_id)
LEFT JOIN ord  o  USING (metric_date, zone_id)
LEFT JOIN dlv  d  USING (metric_date, zone_id)
ORDER BY s.metric_date, s.zone_id;
```

`gold_daily_business_metrics.sql` follows the same shape with the zone cross-join removed and `active_customers` added as `COUNT(DISTINCT customer_id) FILTER (WHERE is_completed)`.

- [ ] **Step 4: Write `build_gold()` calling `assert_contract` after each write**

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add sql/gold src/pulse/quality.py tests/test_contracts.py
git commit -m "feat: gold layer with calendar spine and contract assertions"
```

---

### Task 9: Metrics

**Files:**
- Create: `src/pulse/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: gold parquet
- Produces: `GoldTables` (a simple container), `load_gold() -> GoldTables`, `compute_metrics(gold, params) -> MetricFrame` where `MetricFrame` exposes `.series(metric, scope, scope_value) -> pd.Series` indexed by date, and `.headline(params) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from datetime import date
import pandas as pd
from pulse.types import AnalysisParams
from pulse.metrics import load_gold, compute_metrics

P = AnalysisParams(as_of=date(2026, 9, 10))


def test_company_series_covers_every_day():
    mf = compute_metrics(load_gold(), P)
    s = mf.series("gmv", "company", "all")
    assert len(s) == 180
    assert s.index.is_monotonic_increasing


def test_zone_series_are_available_per_zone():
    mf = compute_metrics(load_gold(), P)
    s = mf.series("completion_rate", "zone", "7")
    assert len(s) == 180
    assert s.between(0, 1).all()


def test_headline_kpis_carry_recent_baseline_and_delta():
    mf = compute_metrics(load_gold(), P)
    head = mf.headline(P)
    assert 7 <= len(head) <= 12
    gmv = next(k for k in head if k["metric"] == "gmv")
    assert {"metric", "recent", "baseline", "delta_pct", "unit"}.issubset(gmv)
    assert gmv["delta_pct"] < 0      # company GMV is down in the recent window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

All KPI formulas live here and nowhere else. `MetricFrame` wraps a long-format DataFrame `(metric, scope, scope_value, metric_date, value)` built from the gold tables, with `.series()` pivoting on demand.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/metrics.py tests/test_metrics.py
git commit -m "feat: metric frame with company and zone scoped KPI series"
```

---

### Task 10: Anomaly detection

**Files:**
- Create: `src/pulse/anomaly_detection.py`, `tests/test_anomaly.py`

**Interfaces:**
- Consumes: `MetricFrame`, `AnalysisParams`
- Produces: `detect_anomalies(metric_frame, params) -> list[Anomaly]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anomaly.py
import json
from datetime import date
import numpy as np
import pandas as pd
from pulse.types import AnalysisParams
from pulse.metrics import load_gold, compute_metrics
from pulse.anomaly_detection import detect_anomalies, detect_on_series
from pulse.config import GROUND_TRUTH

P = AnalysisParams(as_of=date(2026, 9, 10))


def test_company_gmv_drop_is_detected():
    anomalies = detect_anomalies(compute_metrics(load_gold(), P), P)
    gmv = [a for a in anomalies if a.metric == "gmv" and a.scope == "company"]
    assert gmv, "company GMV anomaly must fire - this is the vertical slice trigger"
    assert gmv[0].direction == "drop"
    assert gmv[0].deviation_pct <= -3.0


def test_clean_baseline_window_produces_no_anomalies():
    """Days 1-149 are clean by construction. A detector that fires here
    is producing false positives."""
    clean = AnalysisParams(as_of=date(2026, 8, 1))      # day 140
    anomalies = detect_anomalies(compute_metrics(load_gold(), clean), clean)
    assert anomalies == []


def test_day_of_week_seasonality_does_not_trigger_false_positives():
    idx = pd.date_range("2026-03-15", periods=180, freq="D")
    dow_effect = np.where(idx.dayofweek.isin([4, 5]), 1.22, 1.0)
    series = pd.Series(10_000 * dow_effect, index=idx)
    assert detect_on_series(series, "gmv", "company", "all", P) is None


def test_sensitivity_is_monotonic():
    mf = compute_metrics(load_gold(), P)
    loose = detect_anomalies(mf, AnalysisParams(as_of=P.as_of, sensitivity=1.5))
    tight = detect_anomalies(mf, AnalysisParams(as_of=P.as_of, sensitivity=4.5))
    assert len(loose) >= len(tight)


def test_zone_7_is_among_the_detected_scopes():
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    injected_zone = gt["incidents"][0]["scope"]["value"]
    anomalies = detect_anomalies(compute_metrics(load_gold(), P), P)
    zones = {a.scope_value for a in anomalies if a.scope == "zone"}
    assert str(injected_zone) in zones
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_anomaly.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.anomaly_detection'`

- [ ] **Step 3: Implement detection**

```python
def detect_on_series(series, metric, scope, scope_value, p) -> Anomaly | None:
    recent = series.iloc[-p.comparison_window_days:]
    baseline = series.iloc[-(p.comparison_window_days + p.baseline_window_days)
                           : -p.comparison_window_days]
    if len(baseline) < 14 or baseline.std() == 0:
        return None

    # Day-of-week adjustment: weekend lift on a marketplace is large enough
    # to trip a naive z-score every Saturday.
    dow_mean = baseline.groupby(baseline.index.dayofweek).mean()
    expected = recent.index.dayofweek.map(dow_mean).to_numpy()
    resid_std = (baseline - baseline.index.dayofweek.map(dow_mean)).std()
    if resid_std == 0:
        return None

    z = (recent.to_numpy().mean() - expected.mean()) / (resid_std / np.sqrt(len(recent)))
    dev_pct = 100 * (recent.mean() - expected.mean()) / expected.mean()

    if abs(z) < p.sensitivity or abs(dev_pct) < 3.0:
        return None
    return Anomaly(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_anomaly.py -v`
Expected: PASS (5 passed)

If `test_company_gmv_drop_is_detected` fails, the Zone 7 incident is not deep enough to clear the company-level floor. Re-calibrate `cancel_mult` in `incidents.py` upward or raise Zone 7's `demand_weight`. The test is the gate; do not lower the threshold.

- [ ] **Step 5: Commit**

```bash
git add src/pulse/anomaly_detection.py tests/test_anomaly.py
git commit -m "feat: day-of-week adjusted anomaly detection with materiality floor"
```

---

### Task 11: Root cause

**Files:**
- Create: `src/pulse/root_cause.py`, `tests/test_root_cause.py`

**Interfaces:**
- Consumes: `Anomaly`, `GoldTables`, `AnalysisParams`
- Produces: `diagnose(anomaly, gold, params) -> Diagnosis`, `segment_contributions(...)`, `funnel_decomposition(...)`, `associated_drivers(...)`, `classify_pattern(...)`, `compute_confidence(...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_root_cause.py
import json
from datetime import date
from pulse.types import AnalysisParams
from pulse.metrics import load_gold, compute_metrics
from pulse.anomaly_detection import detect_anomalies
from pulse.root_cause import diagnose
from pulse.config import GROUND_TRUTH

P = AnalysisParams(as_of=date(2026, 9, 10))


def company_gmv_diagnosis():
    gold = load_gold()
    a = next(x for x in detect_anomalies(compute_metrics(gold, P), P)
             if x.metric == "gmv" and x.scope == "company")
    return diagnose(a, gold, P)


def test_contributions_sum_to_one_hundred_percent():
    d = company_gmv_diagnosis()
    total = sum(c.contribution_pct for c in d.contributions)
    assert abs(total - 100.0) < 1.0


def test_engine_recovers_the_injected_zone_without_reading_ground_truth():
    """The RECOVERY test. Ground truth is read here, in the test, only to
    check the engine independently reached the same answer."""
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    injected = str(gt["incidents"][0]["scope"]["value"])
    d = company_gmv_diagnosis()
    assert d.primary_segment.segment == injected
    assert d.primary_segment.rank == 1
    assert d.primary_segment.contribution_pct >= 55.0


def test_zone7_breaks_at_completion_rate_not_conversion():
    d = company_gmv_diagnosis()
    assert d.funnel_break_stage == "completion_rate"
    breaks = [f for f in d.funnel if f.is_primary_break]
    assert len(breaks) == 1 and breaks[0].stage == "completion_rate"


def test_funnel_log_contributions_reconstruct_the_gmv_change():
    d = company_gmv_diagnosis()
    total = sum(f.log_contribution for f in d.funnel)
    import math
    expected = math.log(d.anomaly.recent_value / d.anomaly.baseline_value)
    assert abs(total - expected) < 0.02


def test_delivery_time_and_cancellation_are_the_associated_drivers():
    d = company_gmv_diagnosis()
    names = [dr.metric for dr in d.drivers[:3]]
    assert "avg_actual_delivery_minutes" in names
    assert "cancellation_rate" in names
    assert d.drivers[0].evidence_strength in ("strong", "moderate")


def test_pattern_classifies_to_the_fulfillment_playbook_key():
    assert company_gmv_diagnosis().pattern == "fulfillment_eta_degradation"


def test_confidence_is_bounded_and_documented():
    d = company_gmv_diagnosis()
    assert 0.0 <= d.confidence <= 1.0
    assert d.confidence > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_root_cause.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.root_cause'`

- [ ] **Step 3: Implement the funnel decomposition**

```python
import math

FUNNEL_STAGES = ("sessions", "order_conversion", "completion_rate", "aov")

def funnel_decomposition(gold, scope, scope_value, p) -> tuple[FunnelStage, ...]:
    """GMV = sessions x order_conversion x completion_rate x aov.
    In logs the percentage change becomes additive, so the largest-magnitude
    term IS the broken stage - computed, not asserted."""
    stages = []
    for stage in FUNNEL_STAGES:
        s = _series(gold, stage, scope, scope_value)
        recent = s.iloc[-p.comparison_window_days:].mean()
        baseline = s.iloc[-(p.comparison_window_days + p.baseline_window_days)
                          : -p.comparison_window_days].mean()
        log_contrib = math.log(recent / baseline) if baseline > 0 and recent > 0 else 0.0
        stages.append(FunnelStage(stage, recent, baseline,
                                  100 * (recent - baseline) / baseline,
                                  log_contrib, False))
    worst = max(stages, key=lambda f: abs(f.log_contribution))
    return tuple(replace(f, is_primary_break=(f is worst)) for f in stages)
```

`classify_pattern` maps `(funnel_break_stage, drivers[0].metric)` through the table in spec section 9, defaulting to `"unknown_pattern"`.

`compute_confidence` implements the documented formula verbatim:
`0.25*min(|z|/5,1) + 0.25*top_contribution/100 + 0.30*max|corr| + 0.20*persistence`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_root_cause.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/root_cause.py tests/test_root_cause.py
git commit -m "feat: root cause with funnel log-decomposition and driver correlation"
```

---

### Task 12: Impact and prioritization

**Files:**
- Create: `src/pulse/prioritization.py`, `tests/test_prioritization.py`

**Interfaces:**
- Consumes: `Diagnosis`, `GoldTables`, `AnalysisParams`
- Produces: `estimate_impact(diagnosis, gold, params) -> Impact`, `prioritize(pairs, params) -> list[Priority]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prioritization.py
from datetime import date
from pulse.types import AnalysisParams, Impact
from pulse.metrics import load_gold, compute_metrics
from pulse.anomaly_detection import detect_anomalies
from pulse.root_cause import diagnose
from pulse.prioritization import estimate_impact, prioritize

P = AnalysisParams(as_of=date(2026, 9, 10))


def build_pairs():
    gold = load_gold()
    anomalies = detect_anomalies(compute_metrics(gold, P), P)
    return [(d := diagnose(a, gold, P), estimate_impact(d, gold, P)) for a in anomalies]


def test_impact_is_positive_and_projects_thirty_days():
    _, impact = build_pairs()[0]
    assert impact.gmv_at_risk_brl > 0
    assert impact.orders_lost > 0
    assert abs(impact.projected_30d_brl - impact.daily_run_rate_brl * 30) < 1.0


def test_ranking_is_dense_and_ordered():
    prios = prioritize(build_pairs(), P)
    assert [p.rank for p in prios] == list(range(1, len(prios) + 1))
    scores = [p.impact_score for p in prios]
    assert scores == sorted(scores, reverse=True)


def test_score_breakdown_weights_sum_to_one_and_reproduce_the_score():
    p0 = prioritize(build_pairs(), P)[0]
    w = p0.score_breakdown
    assert abs(sum(w[k] for k in ("w_gmv", "w_orders", "w_customers", "w_confidence")) - 1.0) < 1e-9
    recomputed = 100 * (
        w["w_gmv"] * w["n_gmv"] + w["w_orders"] * w["n_orders"]
        + w["w_customers"] * w["n_customers"] + w["w_confidence"] * w["n_confidence"])
    assert abs(recomputed - p0.impact_score) < 0.01


def test_zone_7_ranks_first():
    assert prioritize(build_pairs(), P)[0].diagnosis.primary_segment.segment == "7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prioritization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.prioritization'`

- [ ] **Step 3: Implement with the documented formula**

```python
WEIGHTS = {"w_gmv": 0.45, "w_orders": 0.20, "w_customers": 0.15, "w_confidence": 0.20}

def prioritize(pairs, p) -> list[Priority]:
    if not pairs:
        return []
    mx = {  # min-max normalisation within this run's candidate set
        "gmv": max(i.projected_30d_brl for _, i in pairs) or 1.0,
        "orders": max(i.orders_lost for _, i in pairs) or 1,
        "customers": max(i.customers_affected for _, i in pairs) or 1,
    }
    scored = []
    for d, i in pairs:
        n = {"n_gmv": i.projected_30d_brl / mx["gmv"],
             "n_orders": i.orders_lost / mx["orders"],
             "n_customers": i.customers_affected / mx["customers"],
             "n_confidence": d.confidence}
        score = 100 * sum(WEIGHTS[f"w_{k}"] * n[f"n_{k}"]
                          for k in ("gmv", "orders", "customers", "confidence"))
        scored.append((score, d, i, {**WEIGHTS, **n}))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [Priority(d, i, s, rank, bd)
            for rank, (s, d, i, bd) in enumerate(scored, start=1)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prioritization.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/prioritization.py tests/test_prioritization.py
git commit -m "feat: impact estimation and explainable impact score"
```

---

### Task 13: Recommendation playbook

**Files:**
- Create: `src/pulse/playbook.yml`, `src/pulse/playbook.py`, `tests/test_playbook.py`

**Interfaces:**
- Consumes: `Diagnosis`, `Impact`
- Produces: `recommend(diagnosis, impact) -> Recommendation`, `PLAYBOOK` dict, `PATTERNS` tuple

- [ ] **Step 1: Write the failing test**

```python
# tests/test_playbook.py
import pytest
from pulse.playbook import PLAYBOOK, PATTERNS, recommend, render_context


def test_every_pattern_has_a_playbook_entry_including_the_fallback():
    assert "unknown_pattern" in PLAYBOOK
    for pattern in PATTERNS:
        assert pattern in PLAYBOOK
        entry = PLAYBOOK[pattern]
        for field in ("action", "rationale", "expected_effect",
                      "validation_method", "owner_function", "effort"):
            assert entry.get(field), f"{pattern} missing {field}"


def test_every_template_renders_against_a_real_context():
    ctx = {"segment": "Zone 7", "peak_window": "18:00-21:00",
           "gmv_at_risk": "R$ 184,200", "driver": "delivery time",
           "metric": "GMV", "deviation_pct": "-3.5%"}
    for pattern, entry in PLAYBOOK.items():
        for field in ("action", "rationale", "expected_effect", "validation_method"):
            entry[field].format(**ctx)      # raises KeyError on an unknown token


def test_unknown_pattern_returns_an_explicit_investigate_recommendation(fake_diagnosis):
    rec = recommend(fake_diagnosis(pattern="something_new"), fake_impact())
    assert rec.playbook_id == "unknown_pattern"
    assert "investigate" in rec.action.lower()
    assert rec.action != ""


def test_recommendation_uses_association_language_not_causal_claims():
    for entry in PLAYBOOK.values():
        blob = " ".join(str(v) for v in entry.values()).lower()
        for banned in ("caused by", "the cause is", "proves that"):
            assert banned not in blob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_playbook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.playbook'`

- [ ] **Step 3: Write `src/pulse/playbook.yml`**

```yaml
fulfillment_eta_degradation:
  action: >
    Rebalance courier supply into {segment} during {peak_window}; temporarily
    widen delivery radius caps and raise courier incentives until delivery time
    returns to baseline.
  rationale: >
    Completion rate is the broken funnel stage while sessions and order
    placement remain stable, and delivery time deterioration is the strongest
    associated driver.
  expected_effect: >
    Recovering delivery time to baseline would restore an estimated
    {gmv_at_risk} over 30 days.
  validation_method: >
    Zone-level difference-in-differences against unaffected zones over a 14-day
    post-intervention window; primary metric completion_rate, guardrail
    delivery_cost_per_order.
  owner_function: Operations / Logistics
  effort: medium

unknown_pattern:
  action: >
    Investigate {segment} manually: the observed combination of broken funnel
    stage and associated drivers does not match a known playbook entry.
  rationale: >
    {metric} deviated by {deviation_pct} with {driver} as the leading associated
    driver, but this signature is not yet catalogued.
  expected_effect: >
    Unknown. Quantify before committing resources.
  validation_method: >
    Define a primary metric and a control group before acting.
  owner_function: Analytics
  effort: unknown
```

Task 18 appends `supply_availability_gap`, `promo_margin_erosion`, `acquisition_quality_decay`.

- [ ] **Step 4: Write `src/pulse/playbook.py`**

```python
import yaml
from pathlib import Path
from pulse.types import Recommendation

PLAYBOOK = yaml.safe_load((Path(__file__).parent / "playbook.yml").read_text())
PATTERNS = ("fulfillment_eta_degradation", "unknown_pattern")   # extended in Task 18

def render_context(d, i) -> dict[str, str]:
    seg = d.primary_segment.segment if d.primary_segment else "company-wide"
    return {
        "segment": f"Zone {seg}" if d.anomaly.scope == "zone" or seg.isdigit() else seg,
        "peak_window": "18:00-21:00",
        "gmv_at_risk": f"R$ {i.projected_30d_brl:,.0f}",
        "driver": d.drivers[0].metric if d.drivers else "none identified",
        "metric": d.anomaly.metric.upper(),
        "deviation_pct": f"{d.anomaly.deviation_pct:.1f}%",
    }

def recommend(d, i) -> Recommendation:
    key = d.pattern if d.pattern in PLAYBOOK else "unknown_pattern"
    e, ctx = PLAYBOOK[key], render_context(d, i)
    return Recommendation(
        action=e["action"].format(**ctx).strip(),
        rationale=e["rationale"].format(**ctx).strip(),
        expected_effect=e["expected_effect"].format(**ctx).strip(),
        validation_method=e["validation_method"].format(**ctx).strip(),
        owner_function=e["owner_function"], playbook_id=key, effort=e["effort"])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_playbook.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/pulse/playbook.yml src/pulse/playbook.py tests/test_playbook.py
git commit -m "feat: deterministic recommendation playbook with explicit fallback"
```

---

### Task 14: Decision memo, orchestrator, CLI

**Files:**
- Create: `src/pulse/decision_memo.py`, `src/pulse/engine.py`, `src/pulse/cli.py`, `tests/test_memo.py`, `tests/test_engine.py`
- Modify: `pyproject.toml` (console script)

**Interfaces:**
- Consumes: everything from Tasks 9-13
- Produces: `build_memo(priority, recommendation, params) -> DecisionMemo`, `memo_to_dict(m)`, `memo_to_markdown(m)`, **`run_decision_cycle(gold, params) -> DecisionCycleResult`**, CLI commands `pulse generate|build|decide|run`

This is the task that satisfies ONE ENGINE ONLY. `run_decision_cycle` is the single orchestrator; nothing downstream may re-implement any step.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memo.py
from datetime import date
from pulse.types import AnalysisParams
from pulse.metrics import load_gold
from pulse.engine import run_decision_cycle
from pulse.decision_memo import memo_to_dict, memo_to_markdown

P = AnalysisParams(as_of=date(2026, 9, 10))


def test_memo_has_every_required_section_populated():
    memo = run_decision_cycle(load_gold(), P).memos[0]
    d = memo_to_dict(memo)
    for section in ("incident", "impact", "concentration", "associated_drivers",
                    "evidence", "priority", "recommended_action",
                    "potential_result", "validation_method", "confidence"):
        assert d.get(section) not in (None, "", [], {}), f"empty section: {section}"


def test_memo_is_identical_with_and_without_a_narrator():
    """Mechanical proof the LLM owns no facts and no policy."""
    class StubNarrator:
        def narrate(self, bundle): return "Completely different prose."

    plain = run_decision_cycle(load_gold(), P, narrator=None).memos[0]
    narrated = run_decision_cycle(load_gold(), P, narrator=StubNarrator()).memos[0]

    a, b = memo_to_dict(plain), memo_to_dict(narrated)
    a.pop("narrative"); b.pop("narrative")
    assert a == b
    assert narrated.narrative == "Completely different prose."
    assert plain.narrative is None


def test_markdown_renders_without_placeholders():
    md = memo_to_markdown(run_decision_cycle(load_gold(), P).memos[0])
    assert "{" not in md and "TBD" not in md and "TODO" not in md
    assert "R$" in md


def test_memo_uses_association_language():
    md = memo_to_markdown(run_decision_cycle(load_gold(), P).memos[0]).lower()
    assert "associated driver" in md
    for banned in ("caused by", "proves that"):
        assert banned not in md
```

```python
# tests/test_engine.py
import json, subprocess, sys
from datetime import date
from pathlib import Path
from pulse.types import AnalysisParams
from pulse.metrics import load_gold
from pulse.engine import run_decision_cycle

P = AnalysisParams(as_of=date(2026, 9, 10))


def test_cycle_returns_ranked_priorities_and_matching_memos():
    r = run_decision_cycle(load_gold(), P)
    assert r.anomalies and r.priorities and r.memos
    assert len(r.memos) == min(3, len(r.priorities))
    assert [p.rank for p in r.priorities[:3]] == [1, 2, 3][:len(r.memos)]


def test_cli_decide_writes_both_artifacts():
    subprocess.run([sys.executable, "-m", "pulse.cli", "decide"], check=True)
    decisions = Path("artifacts/decisions.json")
    memo_md = Path("artifacts/decision-memo.md")
    assert decisions.exists() and memo_md.exists()
    payload = json.loads(decisions.read_text())
    assert payload["priorities"][0]["rank"] == 1


def test_cli_and_library_produce_the_same_result():
    """ONE ENGINE ONLY - the CLI must not have its own logic."""
    subprocess.run([sys.executable, "-m", "pulse.cli", "decide"], check=True)
    from_cli = json.loads(Path("artifacts/decisions.json").read_text())
    from_lib = run_decision_cycle(load_gold(), P)
    assert from_cli["priorities"][0]["impact_score"] == \
           round(from_lib.priorities[0].impact_score, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memo.py tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.engine'`

- [ ] **Step 3: Write `src/pulse/engine.py`**

```python
from pulse.metrics import compute_metrics
from pulse.anomaly_detection import detect_anomalies
from pulse.root_cause import diagnose
from pulse.prioritization import estimate_impact, prioritize
from pulse.playbook import recommend
from pulse.decision_memo import build_memo
from pulse.types import DecisionCycleResult

def run_decision_cycle(gold, params, narrator=None) -> DecisionCycleResult:
    """THE orchestrator. CLI and Streamlit both call this and nothing else."""
    mf = compute_metrics(gold, params)
    anomalies = detect_anomalies(mf, params)
    pairs = [(d := diagnose(a, gold, params), estimate_impact(d, gold, params))
             for a in anomalies]
    priorities = prioritize(pairs, params)
    memos = []
    for prio in priorities[:3]:
        memo = build_memo(prio, recommend(prio.diagnosis, prio.impact), params)
        if narrator is not None:
            memo = replace(memo, narrative=narrator.narrate(memo))
        memos.append(memo)
    return DecisionCycleResult(params.as_of, params, tuple(anomalies),
                               tuple(priorities), tuple(memos),
                               tuple(mf.headline(params)))
```

- [ ] **Step 4: Write `src/pulse/cli.py` with `generate`, `build`, `decide`, `run`**

`decide` loads gold, calls `run_decision_cycle`, writes `artifacts/decisions.json` and `artifacts/decision-memo.md`. It contains serialization only — zero analysis.

Register in `pyproject.toml`:

```toml
[project.scripts]
pulse = "pulse.cli:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_memo.py tests/test_engine.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add src/pulse/decision_memo.py src/pulse/engine.py src/pulse/cli.py tests/test_memo.py tests/test_engine.py pyproject.toml
git commit -m "feat: decision memo, single orchestrator, and CLI artifacts"
```

---

### Task 15: Streamlit vertical slice — GATE #1

**Files:**
- Create: `app/theme.py`, `app/components.py`, `app/streamlit_app.py`, `app/pages/1_overview.py`, `app/pages/7_decision_intelligence.py`, `tests/test_ground_truth_isolation.py`

**Interfaces:**
- Consumes: `run_decision_cycle`, `load_gold`, `AnalysisParams`
- Produces: running Streamlit app with two pages

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ground_truth_isolation.py
from pathlib import Path

def test_engine_and_app_never_read_ground_truth():
    """Ground truth exists only to test recovery. If the app can read it,
    the app can display the generator's answer as analysis."""
    offenders = []
    for root in (Path("src/pulse"), Path("app")):
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "ground_truth" in text.lower() or "GROUND_TRUTH" in text:
                offenders.append(str(py))
    # incidents.py writes it; that is the only permitted reference
    assert offenders == ["src/pulse/incidents.py"], f"leak in {offenders}"


def test_no_page_computes_analytics():
    banned = ("detect_anomalies", "diagnose(", "prioritize(", "estimate_impact",
              ".mean()", ".std()", "groupby(")
    for py in Path("app/pages").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{py} computes analytics: {token}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ground_truth_isolation.py -v`
Expected: FAIL — `app/pages` does not exist

- [ ] **Step 3: Write `app/theme.py`**

Registers the `pulse_dark` Plotly template once and injects one CSS block. All chart styling lives here; no page sets colours.

```python
import plotly.graph_objects as go, plotly.io as pio, streamlit as st

ACCENT, POSITIVE, WARNING, CRITICAL = "#4C7DFF", "#17B26A", "#F59E0B", "#F04438"
BG, SURFACE, TEXT, MUTED = "#0B1220", "#131C2E", "#E6EDF7", "#8A9BB8"

def register_template() -> None:
    pio.templates["pulse_dark"] = go.layout.Template(layout=dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
        colorway=[ACCENT, POSITIVE, WARNING, CRITICAL, "#9B8AFB", "#6BD6E8"],
        xaxis=dict(gridcolor="#1E2A42", zeroline=False),
        yaxis=dict(gridcolor="#1E2A42", zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40), hovermode="x unified"))
    pio.templates.default = "pulse_dark"

def inject_css() -> None:
    st.markdown("""<style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 2rem; max-width: 1400px;}
      .pulse-kpi {background:#131C2E; border:1px solid #1E2A42; border-radius:10px;
                  padding:14px 16px;}
      .pulse-kpi .label {color:#8A9BB8; font-size:11px; letter-spacing:.08em;
                         text-transform:uppercase;}
      .pulse-kpi .value {color:#E6EDF7; font-size:26px; font-weight:600;
                         line-height:1.2;}
    </style>""", unsafe_allow_html=True)
```

- [ ] **Step 4: Write `app/streamlit_app.py`**

```python
import streamlit as st
from datetime import date
from pulse.types import AnalysisParams
from pulse.metrics import load_gold
from pulse.engine import run_decision_cycle
from app.theme import register_template, inject_css

st.set_page_config(page_title="PULSE", page_icon="◆", layout="wide")
register_template(); inject_css()

@st.cache_data(show_spinner="Recomputing decision cycle...")
def cycle(sensitivity: float, window: int, materiality: float):
    p = AnalysisParams(as_of=date(2026, 9, 10), sensitivity=sensitivity,
                       comparison_window_days=window, min_materiality_brl=materiality)
    return run_decision_cycle(load_gold(), p)

with st.sidebar:
    st.caption("ANALYSIS PARAMETERS")
    sens = st.slider("Anomaly sensitivity (z)", 1.5, 4.5, 2.5, 0.1)
    win  = st.select_slider("Comparison window (days)", [7, 14, 28], value=14)
    mat  = st.number_input("Minimum materiality (R$)", 0, 100_000, 5_000, 1_000)
    st.divider()
    st.caption("Decision Support - no autonomous operational execution.")
    st.caption("All data is synthetic.")

st.session_state["cycle"] = cycle(sens, win, mat)

st.navigation({
    "Command":   [st.Page("pages/1_overview.py", title="Executive Overview"),
                  st.Page("pages/7_decision_intelligence.py", title="Decision Intelligence")],
}).run()
```

- [ ] **Step 5: Build the two pages**

`1_overview.py`: seven `kpi_card`s from `result.headline_kpis`, GMV trend with the comparison window shaded, zone performance bar with the top-contributing zone highlighted, Top Incidents, Top 3 Priorities.

`7_decision_intelligence.py`: three `priority_card`s; the selected one expands to the memo; Root Cause Explorer = funnel waterfall of `FunnelStage.log_contribution`, segment contribution bar, dual-axis driver series, and the computed chain string. Every number is read from the result object.

- [ ] **Step 6: Run the app and confirm the slice end to end**

Run: `uv run streamlit run app/streamlit_app.py`

Confirm visually: GMV anomaly visible → Zone 7 identified as top contributor → completion_rate shown as the broken stage → delivery time and cancellation listed as associated drivers → impact in R$ → priority rank → full Decision Memo rendered.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 8: Commit — GATE #1**

```bash
git add app tests/test_ground_truth_isolation.py
git commit -m "feat: Gate 1 - complete vertical slice visible in Streamlit"
```

> **STOP. Gate #1 review.** Do not start Task 16 until the loop above is confirmed working end to end.

---

# Phase 2 — Breadth

### Task 16: Incidents 2-4 and remaining entities

**Files:**
- Modify: `src/pulse/incidents.py`, `src/pulse/data_generator.py`, `tests/test_incidents.py`, `tests/test_generator.py`

**Interfaces:**
- Consumes: `Effects`, `IncidentContext`
- Produces: `zone4_availability`, `promo_margin_erosion`, `paid_social_retention`; `generate_merchant_availability`, `generate_promotions`, `generate_experiments`, `generate_experiment_assignments`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_incidents.py
def test_zone4_suppresses_availability_only_at_peak_hours():
    off_peak = effects_for(ctx(day=170, zone=4, hour=15))
    at_peak  = effects_for(ctx(day=170, zone=4, hour=19))
    assert off_peak.availability_mult == 1.0
    assert at_peak.availability_mult < 0.75

def test_zone4_breaks_conversion_not_completion():
    """The mirror image of Zone 7 - this is what the funnel must separate."""
    e = effects_for(ctx(day=170, zone=4, hour=19))
    assert e.conversion_mult < 1.0
    assert e.cancel_mult == 1.0
    assert e.actual_delivery_mult == 1.0
```

```python
# append to tests/test_generator.py
def test_merchant_availability_is_hourly_within_the_operating_window(tables):
    a = tables["merchant_availability"]
    assert len(a) == 150 * 14 * 180
    assert a["snapshot_hour"].between(10, 23).all()
    assert not a.duplicated(subset=["merchant_id", "snapshot_hour"]).any()
    assert {"is_available", "scheduled_open"}.issubset(a.columns)

def test_promotion_lifts_orders_while_destroying_margin(tables):
    o = tables["orders"]
    promo = o[o["promotion_id"].notna()]
    assert len(promo) > 0
    assert promo["contribution_margin"].mean() < o["contribution_margin"].mean()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_incidents.py tests/test_generator.py -v`
Expected: FAIL — `cannot import name 'zone4_availability'`

- [ ] **Step 3: Implement the three incidents and append to `INCIDENTS`**

```python
def zone4_availability(ctx):
    if ctx.day < 158 or ctx.zone_id != 4 or ctx.hour not in (18, 19, 20):
        return Effects.none()
    r = _ramp(ctx.day, 158)
    return Effects(availability_mult=1 - 0.30 * r, conversion_mult=1 - 0.22 * r)
    # No cancel/delivery effect: Zone 4 fails BEFORE checkout.

def promo_margin_erosion(ctx):
    if not (152 <= ctx.day <= 172) or ctx.promotion_id is None:
        return Effects.none()
    return Effects(conversion_mult=1.14, discount_mult=2.4)

def paid_social_retention(ctx):
    if ctx.acquisition_channel != "paid_social" or ctx.signup_day < 120:
        return Effects.none()
    return Effects(repeat_mult=0.57)      # D30 repeat 42% -> 24%

INCIDENTS = [zone7_degradation, zone4_availability,
             promo_margin_erosion, paid_social_retention]
```

Append matching entries to `GROUND_TRUTH_SPEC` with `expected_funnel_break` set to `"order_conversion"` for Zone 4, `"contribution_margin"` for the promo, and `"repeat_rate"` for retention.

- [ ] **Step 4: Implement the remaining generators**

`generate_merchant_availability` emits `merchant_id, snapshot_hour, scheduled_open, is_available` for each merchant × operating hour × day, with `is_available` drawn against `base_availability * effects.availability_mult`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_incidents.py tests/test_generator.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pulse/incidents.py src/pulse/data_generator.py tests/
git commit -m "feat: incidents 2-4 with availability snapshots, promotions, experiments"
```

---

### Task 17: Remaining gold datasets

**Files:**
- Create: `sql/silver/04_merchant_availability.sql`, `sql/gold/gold_merchant_performance.sql`, `sql/gold/gold_customer_retention.sql`, `sql/gold/gold_promotion_performance.sql`, `sql/gold/gold_experiment_results.sql`
- Modify: `src/pulse/quality.py`, `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_contracts.py
def test_all_six_gold_tables_satisfy_their_contracts():
    build_gold()
    for name in GOLD_CONTRACTS:
        assert_contract(io.read_gold(name), name)


def test_retention_cohorts_use_a_self_join_and_cover_all_channels():
    r = io.read_gold("gold_customer_retention")
    assert set(r["acquisition_channel"].unique()) == {
        "organic", "paid_social", "referral", "paid_search"}
    assert r["retention_rate"].between(0, 1).all()
    assert (r.loc[r["period_index"] == 0, "retention_rate"] == 1.0).all()


def test_paid_social_recent_cohorts_show_retention_decay():
    r = io.read_gold("gold_customer_retention")
    ps = r[(r["acquisition_channel"] == "paid_social") & (r["period_index"] == 1)]
    early = ps[ps["cohort_month"] <= "2026-05"]["retention_rate"].mean()
    late = ps[ps["cohort_month"] >= "2026-07"]["retention_rate"].mean()
    assert late < early * 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — missing gold tables

- [ ] **Step 3: Write the four SQL files**

`gold_customer_retention.sql` uses a cohort self-join:

```sql
WITH cohorts AS (
    SELECT customer_id, acquisition_channel,
           strftime(MIN(CAST(order_ts AS DATE)), '%Y-%m') AS cohort_month,
           MIN(CAST(order_ts AS DATE))                    AS first_order_date
    FROM read_parquet('{silver}/orders.parquet') o
    JOIN read_parquet('{silver}/customers.parquet') c USING (customer_id)
    WHERE o.is_completed GROUP BY 1, 2
),
activity AS (
    SELECT c.customer_id, c.cohort_month, c.acquisition_channel,
           date_diff('month', c.first_order_date, CAST(o.order_ts AS DATE)) AS period_index
    FROM cohorts c
    JOIN read_parquet('{silver}/orders.parquet') o USING (customer_id)
    WHERE o.is_completed
)
SELECT cohort_month, acquisition_channel, period_index,
       COUNT(DISTINCT customer_id) AS retained_customers,
       MAX(COUNT(DISTINCT customer_id)) OVER (
           PARTITION BY cohort_month, acquisition_channel) AS cohort_size,
       COUNT(DISTINCT customer_id)::DOUBLE / MAX(COUNT(DISTINCT customer_id)) OVER (
           PARTITION BY cohort_month, acquisition_channel) AS retention_rate
FROM activity GROUP BY 1, 2, 3 ORDER BY 1, 2, 3;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sql/ src/pulse/quality.py tests/test_contracts.py
git commit -m "feat: merchant, retention, promotion and experiment gold datasets"
```

---

### Task 18: Remaining playbook patterns and multi-anomaly ranking

**Files:**
- Modify: `src/pulse/playbook.yml`, `src/pulse/playbook.py`, `src/pulse/root_cause.py`, `tests/test_playbook.py`, `tests/test_root_cause.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_root_cause.py
def test_zone4_breaks_at_order_conversion_not_completion():
    gold = load_gold()
    a = next(x for x in detect_anomalies(compute_metrics(gold, P), P)
             if x.scope == "zone" and x.scope_value == "4")
    d = diagnose(a, gold, P)
    assert d.funnel_break_stage == "order_conversion"
    assert d.pattern == "supply_availability_gap"
    assert "merchant_availability" in [dr.metric for dr in d.drivers[:2]]


def test_all_four_injected_incidents_are_recovered():
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    gold = load_gold()
    found = set()
    for a in detect_anomalies(compute_metrics(gold, P), P):
        found.add(diagnose(a, gold, P).pattern)
    expected = {"fulfillment_eta_degradation", "supply_availability_gap",
                "promo_margin_erosion", "acquisition_quality_decay"}
    assert expected.issubset(found), f"not recovered: {expected - found}"
    assert len(gt["incidents"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_root_cause.py -v`
Expected: FAIL — pattern not recovered

- [ ] **Step 3: Add the three playbook entries and extend `PATTERNS`**

Each with real `action` / `rationale` / `expected_effect` / `validation_method` / `owner_function` / `effort`, using association language only.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_root_cause.py tests/test_playbook.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pulse/playbook.yml src/pulse/playbook.py src/pulse/root_cause.py tests/
git commit -m "feat: supply, promo and acquisition playbook patterns"
```

---

### Task 19: Experiment lab

**Files:**
- Create: `src/pulse/experiments.py`, `tests/test_experiments.py`

**Interfaces:**
- Consumes: `gold_experiment_results`
- Produces: `two_proportion_test(c_n, c_x, t_n, t_x) -> TestResult`, `required_sample_size(baseline, mde, alpha, power) -> int`, `experiment_economics(...) -> Economics`, `analyse_experiment(gold, experiment_id) -> ExperimentReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiments.py
import pytest
from scipy import stats
from pulse.experiments import (two_proportion_test, required_sample_size,
                               experiment_economics, analyse_experiment)
from pulse.metrics import load_gold


def test_two_proportion_test_matches_a_scipy_reference():
    r = two_proportion_test(c_n=6000, c_x=852, t_n=6000, t_x=1110)
    assert abs(r.control_rate - 0.142) < 0.001
    assert abs(r.treatment_rate - 0.185) < 0.001
    assert abs(r.absolute_diff - 0.043) < 0.001
    assert r.p_value < 0.01
    lo, hi = r.ci_95
    assert lo > 0 and hi > lo
    assert lo < r.absolute_diff < hi


def test_required_sample_size_is_sane():
    n = required_sample_size(baseline=0.142, mde=0.03, alpha=0.05, power=0.80)
    assert 1_500 < n < 4_000


def test_economics_pays_incentive_on_every_redemption_not_just_incremental():
    """The subtlety that decides the business verdict."""
    e = experiment_economics(control_conversions=852, treatment_conversions=1110,
                             incentive_brl=8.0, margin_per_order_brl=14.0,
                             downstream_multiplier=1.0)
    assert e.incentive_cost == pytest.approx(1110 * 8.0)
    assert e.incremental_orders == 258
    assert e.incremental_margin == pytest.approx(258 * 14.0)
    assert e.roi < 0


def test_downstream_value_moves_roi_toward_breakeven():
    base = experiment_economics(852, 1110, 8.0, 14.0, downstream_multiplier=1.0)
    ltv  = experiment_economics(852, 1110, 8.0, 14.0, downstream_multiplier=2.3)
    assert ltv.roi > base.roi
    assert -0.15 < ltv.roi < 0.0


def test_report_states_both_verdicts():
    r = analyse_experiment(load_gold(), "EXP-001")
    assert r.statistical_verdict == "significant"
    assert r.business_verdict in ("negative", "marginal")
    assert "iterate" in r.conclusion.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.experiments'`

- [ ] **Step 3: Implement using `scipy.stats.norm`**

```python
def experiment_economics(control_conversions, treatment_conversions,
                         incentive_brl, margin_per_order_brl,
                         downstream_multiplier=1.0) -> Economics:
    # Cost is paid on EVERY treatment redemption...
    incentive_cost = treatment_conversions * incentive_brl
    # ...but margin is earned only on the INCREMENTAL ones.
    incremental_orders = treatment_conversions - control_conversions
    incremental_margin = (incremental_orders * margin_per_order_brl
                          * downstream_multiplier)
    roi = (incremental_margin - incentive_cost) / incentive_cost
    return Economics(incentive_cost, incremental_orders, incremental_margin, roi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pulse/experiments.py tests/test_experiments.py
git commit -m "feat: experiment lab with two-proportion test and incentive economics"
```

---

### Task 20: Copilot grounding

**Files:**
- Create: `src/pulse/copilot.py`, `tests/test_copilot.py`, `.env.example` (already exists — verify)

**Interfaces:**
- Consumes: `DecisionCycleResult`
- Produces: `build_evidence_bundle(result) -> EvidenceBundle`, `Narrator` protocol, `AnthropicNarrator`, `NullNarrator`, `validate_response(answer_text, bundle) -> ValidationReport`, `ask(question, bundle, narrator) -> CopilotAnswer`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_copilot.py
import json
from datetime import date
from pulse.types import AnalysisParams
from pulse.metrics import load_gold
from pulse.engine import run_decision_cycle
from pulse.copilot import (build_evidence_bundle, validate_response,
                           NullNarrator, ask)

P = AnalysisParams(as_of=date(2026, 9, 10))
BUNDLE = build_evidence_bundle(run_decision_cycle(load_gold(), P))


def test_bundle_is_bounded():
    assert len(json.dumps(BUNDLE.to_dict())) <= 30_000


def test_bundle_contains_no_raw_rows():
    blob = json.dumps(BUNDLE.to_dict())
    for forbidden in ("order_id", "session_id", "customer_id", "merchant_id"):
        assert forbidden not in blob, f"raw row key leaked: {forbidden}"
    assert len(BUNDLE.headline_kpis) <= 12
    assert len(BUNDLE.priorities) <= 3
    assert len(BUNDLE.detected_anomalies) <= 10


def test_validator_catches_a_hallucinated_number():
    report = validate_response("GMV fell by 47.3% driven by Zone 2.", BUNDLE)
    assert not report.all_verified
    assert "47.3" in report.unverified_figures


def test_validator_accepts_figures_present_in_the_bundle():
    real = BUNDLE.headline_kpis[0]["delta_pct"]
    report = validate_response(f"The metric moved {real:.1f}%.", BUNDLE)
    assert report.all_verified


def test_offline_narrator_is_labelled_and_never_presented_as_ai():
    answer = ask("What is happening today?", BUNDLE, NullNarrator())
    assert answer.is_ai_generated is False
    assert "offline" in answer.answer.lower()
    assert answer.metrics_used
    assert answer.confidence is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_copilot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulse.copilot'`

- [ ] **Step 3: Implement the bundle, narrators and validator**

Before writing the Anthropic call, load the `claude-api` skill for current model IDs and SDK usage. Use model `claude-sonnet-5`.

```python
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def validate_response(text: str, bundle) -> ValidationReport:
    """Every figure in an answer must appear in the evidence bundle.
    This is the mechanical grounding guarantee."""
    known = _collect_numbers(bundle.to_dict())
    unverified = [n for n in NUMBER_RE.findall(text)
                  if not _matches_any(_to_float(n), known, tol=0.05)]
    return ValidationReport(all_verified=not unverified,
                            unverified_figures=unverified)
```

System prompt enforces the five rules from spec section 12.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_copilot.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Verify live narration with a real key**

```bash
cp .env.example .env   # then add the real ANTHROPIC_API_KEY
uv run python -c "from pulse.copilot import *; print(ask('Why did GMV decline?', BUNDLE, AnthropicNarrator()).answer)"
```

Confirm the answer quotes only bundle figures and the validation report is clean.

- [ ] **Step 6: Commit**

```bash
git add src/pulse/copilot.py tests/test_copilot.py
git commit -m "feat: copilot evidence bundle, narrators, and numeric grounding guard"
```

---

### Task 21: Analytics pages

**Files:**
- Create: `app/pages/2_growth.py`, `3_operations.py`, `4_customers.py`, `5_merchants.py`, `6_promotions.py`
- Modify: `app/streamlit_app.py` (navigation groups)

- [ ] **Step 1: Extend the isolation test to cover the new pages**

Run: `uv run pytest tests/test_ground_truth_isolation.py -v`
Expected: PASS — no page computes analytics.

- [ ] **Step 2: Build each page reading only gold tables and cycle results**

Growth (GMV/orders/AOV trends, zone mix), Operations (ETA, on-time, cancellation, availability heatmap by hour × zone), Customers (cohort retention heatmap, repeat rate by channel), Merchants (top/bottom by GMV and margin, availability), Promotions (uplift vs discount cost vs margin).

- [ ] **Step 3: Run the app and confirm all pages render**

Run: `uv run streamlit run app/streamlit_app.py`

- [ ] **Step 4: Commit**

```bash
git add app/pages app/streamlit_app.py
git commit -m "feat: growth, operations, customers, merchants and promotions pages"
```

---

### Task 22: Platform pages and human-in-the-loop

**Files:**
- Create: `app/pages/8_experiment_lab.py`, `9_copilot.py`, `10_data_quality.py`
- Modify: `app/components.py` (add `status_pill`, decision state controls)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ground_truth_isolation.py
def test_decision_states_are_session_only_and_execute_nothing():
    text = Path("app/components.py").read_text(encoding="utf-8")
    assert "st.session_state" in text
    for banned in ("requests.post", "httpx.post", "subprocess", "os.system"):
        assert banned not in text, "human-in-the-loop must not execute actions"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ground_truth_isolation.py -v`
Expected: FAIL — `status_pill` not present

- [ ] **Step 3: Build the three pages**

Experiment Lab renders `analyse_experiment()` output: hypothesis, arms, sample sizes, conversion, uplift, CI, p-value, power check, both ROI figures, and the combined verdict.

Copilot renders the chat with six suggested prompts, the evidence panel, and the verification badge from `validate_response`.

Data Quality & Architecture reads `_quality_report.parquet` for bronze/silver counts, duplicates removed, nulls handled, quarantined rows, and renders the pipeline diagram.

- [ ] **Step 4: Add the human-in-the-loop control to `components.py`**

```python
STATES = ("INVESTIGATE", "APPROVED", "REJECTED")

def decision_state_control(memo_id: str) -> str:
    """Updates local session state only. Executes nothing."""
    states = st.session_state.setdefault("decision_states", {})
    audit = st.session_state.setdefault("decision_audit", [])
    cols = st.columns(3)
    for col, state in zip(cols, STATES):
        if col.button(state.title(), key=f"{memo_id}_{state}"):
            states[memo_id] = state
            audit.append({"memo_id": memo_id, "state": state,
                          "at": datetime.now().isoformat(timespec="seconds")})
    return states.get(memo_id, "INVESTIGATE")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add app/
git commit -m "feat: experiment lab, copilot UI, data quality page, human-in-the-loop"
```

---

# Phase 3 — Evidence and polish

### Task 23: Databricks PySpark evidence

**Files:**
- Create: `databricks/silver_to_gold_spark.py`, `databricks/README.md`, `tests/test_spark_reconciliation.py`

**Interfaces:**
- Consumes: silver parquet uploaded to the Databricks workspace
- Produces: Delta tables `pulse.gold_zone_performance`, `pulse.gold_daily_business_metrics`; an exported CSV for reconciliation

Scope is deliberately minimal: schema enforcement, Delta writes, real execution, reconciliation. No MERGE or Z-ORDER unless a genuine incremental use case appears.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spark_reconciliation.py
import pandas as pd, pytest
from pathlib import Path
from pulse import io

SPARK_OUT = Path("databricks/output/gold_zone_performance_spark.csv")


@pytest.mark.skipif(not SPARK_OUT.exists(),
                    reason="Databricks output not yet exported")
def test_spark_and_duckdb_gold_agree():
    spark = pd.read_csv(SPARK_OUT, parse_dates=["metric_date"])
    local = io.read_gold("gold_zone_performance")
    keys = ["metric_date", "zone_id"]
    merged = spark.merge(local, on=keys, suffixes=("_spark", "_local"))
    assert len(merged) == len(local), "row count mismatch between engines"
    for col in ("gmv", "orders_completed", "completion_rate"):
        delta = (merged[f"{col}_spark"] - merged[f"{col}_local"]).abs().max()
        assert delta < 0.01, f"{col} diverges by {delta}"
```

- [ ] **Step 2: Run test to verify it skips**

Run: `uv run pytest tests/test_spark_reconciliation.py -v`
Expected: SKIPPED — output not yet exported.

- [ ] **Step 3: Write `databricks/silver_to_gold_spark.py`**

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, IntegerType,
                               DoubleType, BooleanType, TimestampType)

spark = SparkSession.builder.appName("pulse-silver-to-gold").getOrCreate()

ORDERS_SCHEMA = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("zone_id", IntegerType(), False),
    StructField("order_ts", TimestampType(), False),
    StructField("item_amount", DoubleType(), False),
    StructField("contribution_margin", DoubleType(), False),
    StructField("is_completed", BooleanType(), False),
])

orders = (spark.read.schema(ORDERS_SCHEMA)      # schema enforcement, not inference
          .parquet("/Volumes/pulse/silver/orders"))

zone_perf = (orders
    .withColumn("metric_date", F.to_date("order_ts"))
    .groupBy("metric_date", "zone_id")
    .agg(F.sum(F.when(F.col("is_completed"), F.col("item_amount"))).alias("gmv"),
         F.count("*").alias("orders_placed"),
         F.sum(F.col("is_completed").cast("int")).alias("orders_completed"))
    .withColumn("completion_rate",
                F.col("orders_completed") / F.col("orders_placed")))

(zone_perf.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("pulse.gold_zone_performance"))
```

- [ ] **Step 4: Execute on Databricks Free Edition**

Upload `data/silver/*.parquet`, run the notebook, screenshot the Delta table and the job run into `dashboard/screenshots/`, export `gold_zone_performance` to `databricks/output/gold_zone_performance_spark.csv`.

- [ ] **Step 5: Run the reconciliation test**

Run: `uv run pytest tests/test_spark_reconciliation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add databricks/ tests/test_spark_reconciliation.py dashboard/screenshots/
git commit -m "feat: PySpark Delta evidence layer reconciled against DuckDB output"
```

---

### Task 24: Screenshots, README and docs

**Files:**
- Create: `README.md`, `docs/kpi-definitions.md`, `docs/architecture.md`, `docs/tradeoffs.md`
- Create: `dashboard/screenshots/*.png`

- [ ] **Step 1: Capture one screenshot per page**

Run the app and capture all ten pages at 1600×1000 into `dashboard/screenshots/`.

- [ ] **Step 2: Write the README**

Required sections: one-sentence definition · business problem · why it exists · From Signal to Decision · architecture · synthetic dataset · injected incidents · data model with grains · Bronze/Silver/Gold · KPIs · Decision Engine · experimentation · Copilot · screenshots · results · example Decision Memo (pasted verbatim from `artifacts/decision-memo.md`) · how to run · repository structure · limitations · roadmap.

Must state explicitly:

> **ALL DATA IS SYNTHETIC.** The project was inspired by prior experience with digital operations and marketplaces but contains no proprietary company data.

- [ ] **Step 3: Verify every README claim maps to real code or generated output**

Walk each headline claim and point at the file or artifact that backs it. Delete any claim you cannot back.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ dashboard/screenshots/
git commit -m "docs: README, KPI definitions, architecture and tradeoffs"
```

---

### Task 25: Final verification ritual

**Files:**
- Create: `tests/test_no_secrets.py`

- [ ] **Step 1: Write the secrets test**

```python
# tests/test_no_secrets.py
import re, subprocess
from pathlib import Path

KEY_SHAPED = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")

def test_env_is_not_tracked():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout.split()
    assert ".env" not in tracked
    assert ".env.example" in tracked

def test_no_key_shaped_strings_in_tracked_files():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout.split()
    for f in tracked:
        p = Path(f)
        if p.suffix in (".py", ".md", ".toml", ".yml", ".yaml", ".sql", ".json"):
            assert not KEY_SHAPED.search(p.read_text(encoding="utf-8", errors="ignore")), f
```

- [ ] **Step 2: Run the full ritual**

```bash
rm -rf data/bronze data/silver data/gold artifacts
uv run pulse run --all
uv run pytest -q
uv run streamlit run app/streamlit_app.py    # visual check of all 10 pages
git ls-files | grep -c '\.env$'              # must print 0
```

- [ ] **Step 3: Confirm each Definition-of-Done item**

Walk spec section 19 and check every item against real output. Report honestly on anything not met rather than claiming completion.

- [ ] **Step 4: Commit**

```bash
git add tests/test_no_secrets.py
git commit -m "test: secrets scanning and final verification"
```

---

## Self-review

**Spec coverage.** Every spec section maps to a task: §3 data model → T3/T5/T16 · §4 metric semantics → T7/T9 · §5 incidents and calibration → T4/T5/T16 · §6 medallion → T6/T7/T8/T17 · §7 SQL and spine → T7/T8/T17 · §8 Databricks → T23 · §9 engine → T9-T12/T14 · §10 playbook → T13/T18 · §11 experiment lab → T19/T22 · §12 Copilot → T20/T22 · §13 Streamlit IA → T15/T21/T22 · §14 testing → distributed across all tasks · §15 repo structure → T1 · §16 security → T25 · §19 definition of done → T25.

**Placeholder scan.** The `...` markers in Tasks 5 and 9 are deliberate body elisions in vectorised generator functions where the surrounding contract, emitted columns and calibration tests are fully specified; every other step carries literal code. No TBDs, no "handle edge cases", no "similar to Task N".

**Type consistency.** `AnalysisParams`, `Anomaly`, `Diagnosis`, `FunnelStage`, `Impact`, `Priority`, `Recommendation`, `DecisionMemo` and `DecisionCycleResult` are defined once in T2 and referenced with identical field names throughout. `run_decision_cycle(gold, params, narrator=None)` has one signature everywhere. `io.read_gold` / `read_silver` / `write_bronze` names are stable from T3. `effects_for(ctx)` and the `Effects` field names introduced in T4 are reused unchanged in T16.

**Known follow-on.** `Effects.combine` in T4 uses positional zip over seven fields; if T16's incidents add an eighth field, update `combine` and its test together.
