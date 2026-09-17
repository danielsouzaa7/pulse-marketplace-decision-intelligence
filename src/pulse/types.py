from dataclasses import dataclass
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
    # Is customers_affected a MEASURED count or a PRIORITISATION RESIDUAL?
    #
    # False: a distinct count of the customers who ordered in this scope over
    # the comparison window, straight out of silver.
    # True: that count less the counts the nested segment-scope groups in the
    # same run already claim -- a quantity that exists so the ranking does not
    # weight the same customers at two levels. It is smaller than the scope's
    # measured population (5,632 measured, 3,896 residual on this dataset) and
    # is NOT a count of anybody: no 3,896 customers can be enumerated.
    #
    # The flag exists because the number alone cannot say which it is, and the
    # two were being rendered under one label. prioritization._net_of_nested is
    # the only place that sets it True.
    customers_are_residual: bool = False


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
    # The same window-mean KPI rows as headline_kpis, measured at the scope of
    # each priority that got a memo, and carrying scope/scope_value so one flat
    # tuple covers them all. A company dashboard is exactly where a segment
    # incident hides -- zone 7's orders PLACED rose while its orders COMPLETED
    # fell, and no company row shows that. Computed once here so a rendering
    # surface can show it without becoming a second engine.
    segment_kpis: tuple[dict, ...] = ()
