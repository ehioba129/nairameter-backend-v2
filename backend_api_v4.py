"""
NairaMeter Backend API — v4 (Real Database)
=============================================
Every endpoint now queries your live PostgreSQL/TimescaleDB database instead
of reading meter_readings_daily.csv directly. This is the final step: data
now persists properly, survives restarts, and reflects schema.sql's actual
design (readings and confirmed incidents live in separate tables).

Honest architectural note (read this before wondering why a chart changed):
  Real meters report kwh_recorded only — there is no "kwh_true" in
  production, since no meter can measure the consumption a theft victim
  hides. The old CSV-based version showed "Recorded vs. True" because that
  was a synthetic dataset with an artificial ground-truth column. Now that
  we're on the real schema, the Portfolio chart shows recorded consumption
  alone, and confirmed theft comes from the 'incidents' table (populated by
  migrate_incidents.py) rather than a per-row flag.

Setup:
    pip install fastapi uvicorn pydantic psycopg2-binary lightgbm shap joblib pandas
    Set DATABASE_URL, then:
    python -m uvicorn backend_api_v4:app --reload --port 8000

Requires: schema.sql already run, meter_readings loaded via ingestion_service.py,
incidents loaded via migrate_incidents.py, and model_artifacts/ from train_model.py.
"""

import json
import os
import math
from urllib.parse import quote as urlquote
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train_model import engineer_features, encode_categoricals, describe_top_factors, factors_to_sentence

DATABASE_URL = os.environ.get("DATABASE_URL")

# Fallback: build the connection string from separate simple variables if
# DATABASE_URL itself isn't set — much harder to get wrong than one long
# combined string, since each piece is short and easy to verify individually.
if not DATABASE_URL:
    _host = os.environ.get("DB_HOST")
    _port = os.environ.get("DB_PORT", "5432")
    _name = os.environ.get("DB_NAME")
    _user = os.environ.get("DB_USER")
    _password = os.environ.get("DB_PASSWORD")
    if _host and _name and _user and _password:
        # URL-encode user/password: special characters like '!', '@', '#' are
        # valid in a real password but break a connection URI if left raw —
        # this is exactly the bug that caused the "password authentication
        # failed" errors when a password containing '!' was used unencoded.
        DATABASE_URL = f"postgres://{urlquote(_user)}:{urlquote(_password)}@{_host}:{_port}/{_name}?sslmode=require"
MODEL_DIR = os.environ.get("NAIRAMETER_MODEL_DIR", "model_artifacts")

app = FastAPI(title="NairaMeter API", description="Database-backed revenue-protection API.", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["*"])

# =====================================================================
# API KEY PROTECTION — now scoped per-partner, not a single shared secret.
#
# Each key maps to a "scope": either None (full access — for your own
# internal team) or a specific partner name (e.g. "cesel"), which HARD-LOCKS
# that key to only ever see that partner's data — enforced here on the
# server, not left to a client-side filter the user could simply change.
# This closes a real gap: previously, any holder of the single shared key
# could select "All Data" or another partner's name from the dashboard's
# own dropdown and see everything, including data belonging to a different
# partner than the one they were given access for.
#
# Configure via environment variables on Render:
#   API_KEY            -> full access (your internal team)
#   CESEL_API_KEY       -> locked to partner "cesel"
#   HUSK_API_KEY        -> locked to partner "husk_power"
# Add more PARTNERNAME_API_KEY variables the same way as you onboard
# further partners — no code change needed, just a new env var plus one
# line in PARTNER_KEY_ENV below.
# =====================================================================
API_KEY = os.environ.get("API_KEY")  # kept for the startup warning below

PARTNER_KEY_ENV = {
    "cesel": "CESEL_API_KEY",
    "husk_power": "HUSK_API_KEY",
}

def _build_key_scope_map():
    """Builds {api_key_value: partner_scope_or_None} from environment variables."""
    mapping = {}
    if API_KEY:
        mapping[API_KEY] = None  # None = unrestricted, full-access key
    for partner, env_name in PARTNER_KEY_ENV.items():
        partner_key = os.environ.get(env_name)
        if partner_key:
            mapping[partner_key] = partner
    return mapping

KEY_SCOPE_MAP = _build_key_scope_map()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str = Security(api_key_header)) -> Optional[str]:
    """
    Returns the caller's enforced partner scope: None for a full-access key,
    or a partner name (e.g. "cesel") for a partner-restricted key. Endpoints
    use this returned scope to filter data — not whatever the client's own
    query parameters ask for — so a restricted key can never see another
    partner's data no matter what the frontend requests.
    """
    if KEY_SCOPE_MAP and provided_key not in KEY_SCOPE_MAP:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return KEY_SCOPE_MAP.get(provided_key)  # None if key-checking is disabled entirely


def resolve_partner(requested_partner: Optional[str], enforced_scope: Optional[str]) -> Optional[str]:
    """
    The single rule every endpoint uses to decide which partner to filter by:
    - A restricted key's scope always wins, regardless of what was requested.
    - A full-access key may request any partner, or none (see everything).
    """
    return enforced_scope if enforced_scope else requested_partner


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL)


def safe_float(val, ndigits=None):
    """
    Converts a value that may be a pandas NaN (how a SQL NULL/empty SUM or
    AVG surfaces after a database round-trip) into a genuine Python None.
    A plain 'is not None' check does NOT catch this — NaN is not None, but
    it is also not valid JSON, so it must be converted explicitly or the
    response fails to parse in the browser. Use this anywhere a database
    aggregate (SUM, AVG) might legitimately have no rows to aggregate.
    """
    try:
        if val is None or math.isnan(val):
            return None
    except TypeError:
        pass
    return round(float(val), ndigits) if ndigits is not None else float(val)


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Runs a query and returns a DataFrame — the simplest reliable bridge
    between psycopg2 and pandas for read-heavy endpoints like these."""
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params)


# =====================================================================
# MODEL LOADING — same as v3, unchanged
# =====================================================================
class ModelBundle:
    model = None
    explainer = None
    encoders = None
    feature_cols = None
    threshold = 0.5
    loaded = False


BUNDLE = ModelBundle()


@app.on_event("startup")
def load_model():
    try:
        import joblib
        BUNDLE.model = joblib.load(os.path.join(MODEL_DIR, "lightgbm_theft_model.joblib"))
        BUNDLE.encoders = joblib.load(os.path.join(MODEL_DIR, "categorical_encoders.joblib"))
        with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
            BUNDLE.feature_cols = json.load(f)
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            BUNDLE.threshold = json.load(f).get("decision_threshold", 0.5)
        import shap
        BUNDLE.explainer = shap.TreeExplainer(BUNDLE.model)
        BUNDLE.loaded = True
        print(f"Model loaded. Decision threshold: {BUNDLE.threshold:.3f}")
    except FileNotFoundError:
        print(f"WARNING: model_artifacts not found in '{MODEL_DIR}/'. /api/alerts will report 503 until trained.")

    if not DATABASE_URL:
        print("WARNING: DATABASE_URL is not set. All endpoints will fail until it's configured.")

    if not KEY_SCOPE_MAP:
        print("WARNING: no API keys are configured — this API is currently PUBLIC with no protection. "
              "Set the API_KEY environment variable (and optional PARTNERNAME_API_KEY variables) to require X-API-Key on every request.")
    else:
        n_partner_keys = sum(1 for v in KEY_SCOPE_MAP.values() if v is not None)
        print(f"API key protection is ACTIVE — 1 full-access key, {n_partner_keys} partner-scoped key(s).")


# =====================================================================
# RESPONSE SCHEMAS
# =====================================================================
class KPISummary(BaseModel):
    total_customers: int
    n_clusters: int
    customers_with_theft: int
    avg_power_factor: float
    tamper_events: int
    outage_events: int
    model_loaded: bool


class TimeseriesPoint(BaseModel):
    date: date
    kwh_recorded: float


class ClusterSummaryRow(BaseModel):
    cluster_id: str
    n_customers: int
    incident_count: int
    customers_with_theft: int
    avg_power_factor: float


class CustomerListItem(BaseModel):
    customer_id: str
    cluster_id: str
    customer_type: str
    is_theft_customer: bool


class CustomerDailyPoint(BaseModel):
    date: date
    voltage: Optional[float] = None
    current: Optional[float] = None
    active_power_kw: Optional[float] = None
    power_factor: Optional[float] = None
    kwh_recorded: Optional[float] = None
    is_theft: int
    is_outage: int
    tamper_flag: int


class TheftTypeCount(BaseModel):
    theft_type: str
    count: int


class TheftCase(BaseModel):
    customer_id: str
    cluster_id: str
    customer_type: str
    theft_types: List[str]
    theft_days: int
    kwh_stolen: Optional[float] = None
    first_date: date
    last_date: date


class CaseStatus(BaseModel):
    customer_id: str
    status: str
    updated_at: Optional[str] = None


class CaseStatusUpdate(BaseModel):
    status: str  # must be one of: open, investigating, confirmed, false_positive


class Alert(BaseModel):
    alert_id: str
    customer_id: str
    cluster_id: str
    timestamp: date
    severity: str
    confidence: float
    explanation: str
    top_factors: List[str]


# =====================================================================
# ENDPOINTS — every one now a real SQL query against the live database
# =====================================================================
@app.get("/api/kpis", response_model=KPISummary)
def kpis(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys — their own scope always applies."),
          enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "WHERE source_partner = %s" if partner else ""
    params = (partner,) if partner else ()

    total_customers = query_df(f"SELECT COUNT(*) AS n FROM offtakers {partner_filter}", params).iloc[0]["n"]
    n_clusters = query_df(f"SELECT COUNT(DISTINCT cluster_id) AS n FROM offtakers {partner_filter}", params).iloc[0]["n"]

    incident_filter = "WHERE confirmed = true AND meter_id IN (SELECT offtaker_id FROM offtakers WHERE source_partner = %s)" if partner else "WHERE confirmed = true"
    customers_with_theft = query_df(
        f"SELECT COUNT(DISTINCT meter_id) AS n FROM incidents {incident_filter}", params
    ).iloc[0]["n"]

    reading_filter = "WHERE meter_id IN (SELECT offtaker_id FROM offtakers WHERE source_partner = %s)" if partner else ""
    stats = query_df(
        f"SELECT AVG(power_factor) AS avg_pf, SUM(tamper_status::int) AS tamper, SUM(is_outage::int) AS outage "
        f"FROM meter_readings {reading_filter}", params
    ).iloc[0]
    return KPISummary(
        total_customers=int(total_customers), n_clusters=int(n_clusters),
        customers_with_theft=int(customers_with_theft),
        avg_power_factor=round(float(stats["avg_pf"]), 3) if stats["avg_pf"] is not None else 0.0,
        tamper_events=int(stats["tamper"] or 0), outage_events=int(stats["outage"] or 0),
        model_loaded=BUNDLE.loaded,
    )


@app.get("/api/portfolio/timeseries", response_model=List[TimeseriesPoint])
def portfolio_timeseries(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
                          enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "WHERE meter_id IN (SELECT offtaker_id FROM offtakers WHERE source_partner = %s)" if partner else ""
    params = (partner,) if partner else ()
    df = query_df(
        f"SELECT reading_time::date AS date, SUM(kwh_recorded) AS kwh_recorded "
        f"FROM meter_readings {partner_filter} GROUP BY reading_time::date ORDER BY date", params
    )
    return [TimeseriesPoint(date=r["date"], kwh_recorded=round(r["kwh_recorded"], 1)) for _, r in df.iterrows()]


@app.get("/api/clusters/summary", response_model=List[ClusterSummaryRow])
def cluster_summary(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
                     enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "WHERE o.source_partner = %s" if partner else ""
    params = (partner,) if partner else ()
    df = query_df(f"""
        SELECT o.cluster_id,
               COUNT(DISTINCT o.offtaker_id) AS n_customers,
               COUNT(i.incident_id) AS incident_count,
               COUNT(DISTINCT i.meter_id) AS customers_with_theft,
               AVG(mr.power_factor) AS avg_power_factor
        FROM offtakers o
        LEFT JOIN meter_readings mr ON mr.meter_id = o.offtaker_id
        LEFT JOIN incidents i ON i.meter_id = o.offtaker_id AND i.confirmed = true
        {partner_filter}
        GROUP BY o.cluster_id
        ORDER BY customers_with_theft DESC
    """, params)
    return [ClusterSummaryRow(
        cluster_id=r["cluster_id"], n_customers=int(r["n_customers"]),
        incident_count=int(r["incident_count"]), customers_with_theft=int(r["customers_with_theft"]),
        avg_power_factor=safe_float(r["avg_power_factor"], 3) or 0.0,
    ) for _, r in df.iterrows()]


@app.get("/api/customers", response_model=List[CustomerListItem])
def customer_list(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
                   enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "WHERE o.source_partner = %s" if partner else ""
    params = (partner,) if partner else ()
    df = query_df(f"""
        SELECT o.offtaker_id AS customer_id, o.cluster_id, o.customer_type,
               EXISTS(SELECT 1 FROM incidents i WHERE i.meter_id = o.offtaker_id AND i.confirmed = true) AS is_theft
        FROM offtakers o {partner_filter} ORDER BY o.offtaker_id
    """, params)
    return [CustomerListItem(customer_id=r["customer_id"], cluster_id=r["cluster_id"],
                              customer_type=r["customer_type"], is_theft_customer=bool(r["is_theft"]))
            for _, r in df.iterrows()]


@app.get("/api/customers/{customer_id}/detail", response_model=List[CustomerDailyPoint])
def customer_detail(customer_id: str, enforced_scope: Optional[str] = Depends(require_api_key)):
    if enforced_scope:
        owner = query_df("SELECT source_partner FROM offtakers WHERE offtaker_id = %s", (customer_id,))
        if owner.empty or owner.iloc[0]["source_partner"] != enforced_scope:
            # 404, not 403 — a restricted key shouldn't learn that a customer
            # ID exists at all if it belongs to a different partner.
            raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    df = query_df("""
        SELECT mr.reading_time::date AS date, mr.voltage, mr.current, mr.active_power_kw,
               mr.power_factor, mr.kwh_recorded, mr.tamper_status, mr.is_outage,
               (i.incident_id IS NOT NULL) AS is_theft
        FROM meter_readings mr
        LEFT JOIN incidents i ON i.meter_id = mr.meter_id AND i.incident_date = mr.reading_time::date AND i.confirmed = true
        WHERE mr.meter_id = %s
        ORDER BY mr.reading_time
    """, (customer_id,))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    def nan_to_none(val):
        """Pandas represents a database NULL as float NaN, not Python None.
        NaN is technically still a valid float, so it slips past an Optional[float]
        type check — but NaN cannot be represented in strict JSON (it serializes
        as the bareword NaN, not null), which browsers correctly reject as invalid,
        causing exactly the 'failed to load' error seen on real CESEL meters with
        gaps in their data. Must convert to true None before returning."""
        try:
            return None if (val is None or math.isnan(val)) else val
        except TypeError:
            return val

    return [CustomerDailyPoint(
        date=r["date"], voltage=nan_to_none(r["voltage"]), current=nan_to_none(r["current"]),
        active_power_kw=nan_to_none(r["active_power_kw"]), power_factor=nan_to_none(r["power_factor"]),
        kwh_recorded=nan_to_none(r["kwh_recorded"]),
        is_theft=int(r["is_theft"]), is_outage=int(r["is_outage"]), tamper_flag=int(r["tamper_status"]),
    ) for _, r in df.iterrows()]


@app.get("/api/theft/types", response_model=List[TheftTypeCount])
def theft_types(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
                 enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "AND meter_id IN (SELECT offtaker_id FROM offtakers WHERE source_partner = %s)" if partner else ""
    params = (partner,) if partner else ()
    df = query_df(
        f"SELECT incident_type AS theft_type, COUNT(*) AS count FROM incidents "
        f"WHERE confirmed = true {partner_filter} GROUP BY incident_type ORDER BY count DESC", params
    )
    return [TheftTypeCount(theft_type=r["theft_type"], count=int(r["count"])) for _, r in df.iterrows()]


@app.get("/api/theft/cases", response_model=List[TheftCase])
def theft_cases(partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
                 enforced_scope: Optional[str] = Depends(require_api_key)):
    partner = resolve_partner(partner, enforced_scope)
    partner_filter = "AND o.source_partner = %s" if partner else ""
    params = (partner,) if partner else ()
    df = query_df(f"""
        SELECT i.meter_id AS customer_id, o.cluster_id, o.customer_type,
               array_agg(DISTINCT i.incident_type) AS theft_types,
               COUNT(*) AS theft_days, SUM(i.kwh_stolen_est) AS kwh_stolen,
               MIN(i.incident_date) AS first_date, MAX(i.incident_date) AS last_date
        FROM incidents i
        JOIN offtakers o ON o.offtaker_id = i.meter_id
        WHERE i.confirmed = true {partner_filter}
        GROUP BY i.meter_id, o.cluster_id, o.customer_type
        ORDER BY kwh_stolen DESC
    """, params)
    return [TheftCase(
        customer_id=r["customer_id"], cluster_id=r["cluster_id"], customer_type=r["customer_type"],
        theft_types=sorted(r["theft_types"]), theft_days=int(r["theft_days"]),
        kwh_stolen=safe_float(r["kwh_stolen"], 2), first_date=r["first_date"], last_date=r["last_date"],
    ) for _, r in df.iterrows()]


VALID_STATUSES = {"open", "investigating", "confirmed", "false_positive"}


@app.get("/api/cases/status", response_model=List[CaseStatus])
def get_case_statuses(enforced_scope: Optional[str] = Depends(require_api_key)):
    """
    Returns the current investigative status for every case that has one set.
    A customer with no row here is implicitly 'open' — the dashboard treats
    a missing entry as the default, so this only needs to return overrides.
    """
    partner_filter = "WHERE meter_id IN (SELECT offtaker_id FROM offtakers WHERE source_partner = %s)" if enforced_scope else ""
    params = (enforced_scope,) if enforced_scope else ()
    df = query_df(f"SELECT meter_id AS customer_id, status, updated_at::text FROM case_status {partner_filter}", params)
    return [CaseStatus(customer_id=r["customer_id"], status=r["status"], updated_at=r["updated_at"])
            for _, r in df.iterrows()]


@app.put("/api/cases/{customer_id}/status")
def set_case_status(customer_id: str, body: CaseStatusUpdate, enforced_scope: Optional[str] = Depends(require_api_key)):
    """
    Sets (upserts) the investigative status for one case. This is what makes
    status changes persistent and shared across every device/user, replacing
    the old browser-only localStorage approach.
    """
    if body.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {sorted(VALID_STATUSES)}")

    if enforced_scope:
        owner = query_df("SELECT source_partner FROM offtakers WHERE offtaker_id = %s", (customer_id,))
        if owner.empty or owner.iloc[0]["source_partner"] != enforced_scope:
            raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO case_status (meter_id, status, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (meter_id) DO UPDATE SET status = EXCLUDED.status, updated_at = now();
            """, (customer_id, body.status))
        conn.commit()
    return {"customer_id": customer_id, "status": body.status}



@app.get("/api/alerts", response_model=List[Alert])
def alerts(cluster_id: Optional[str] = Query(None), min_confidence: float = Query(0.0, ge=0.0, le=1.0),
           partner: Optional[str] = Query(None, description="Ignored for partner-restricted keys."),
           enforced_scope: Optional[str] = Depends(require_api_key)):
    """
    Scores real readings from the database through your trained model. Column
    names are aliased in SQL to match exactly what engineer_features() in
    train_model.py expects (meter_id -> customer_id, reading_time -> date,
    tamper_status -> tamper_flag), so the identical training-time pipeline
    runs unchanged here.
    """
    if not BUNDLE.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    partner = resolve_partner(partner, enforced_scope)
    conditions = []
    params = []
    if cluster_id:
        conditions.append("o.cluster_id = %s")
        params.append(cluster_id)
    if partner:
        conditions.append("o.source_partner = %s")
        params.append(partner)
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params = tuple(params)
    df = query_df(f"""
        SELECT mr.meter_id AS customer_id, mr.reading_time::date AS date,
               mr.voltage, mr.current, mr.active_power_kw, mr.reactive_power_kvar,
               mr.power_factor, mr.kwh_recorded, mr.tamper_status AS tamper_flag,
               mr.is_outage, o.cluster_id, o.customer_type
        FROM meter_readings mr
        JOIN offtakers o ON o.offtaker_id = mr.meter_id
        {where_clause}
        ORDER BY mr.meter_id, mr.reading_time
    """, params)

    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"])
    feat_df = engineer_features(df)
    feat_df, _ = encode_categoricals(feat_df, encoders=BUNDLE.encoders)
    X = feat_df[BUNDLE.feature_cols]

    # Score everything (cheap — this is just the trained tree ensemble, fast even on
    # the full dataset), but only compute SHAP explanations for rows that actually
    # get flagged. SHAP is the expensive step; running it on 90,000+ rows to
    # explain the ~50 that matter was the main cause of slow load times here.
    proba = BUNDLE.model.predict_proba(X)[:, 1]
    threshold = max(min_confidence, BUNDLE.threshold)
    flagged_idx = np.where((proba >= threshold) | (feat_df["tamper_flag"].values == 1))[0]

    # Cap how many rows we ever explain in one request — protects against a
    # pathological case (e.g., threshold set to 0) still trying to SHAP-explain
    # thousands of rows at once.
    flagged_idx = flagged_idx[:300]

    if len(flagged_idx) > 0:
        shap_theft_subset = BUNDLE.explainer.shap_values(X.iloc[flagged_idx])
        shap_theft_subset = shap_theft_subset[1] if isinstance(shap_theft_subset, list) else shap_theft_subset
    else:
        shap_theft_subset = []

    results = []
    for local_i, i in enumerate(flagged_idx):
        row = feat_df.iloc[i]
        top_factors = describe_top_factors(shap_theft_subset[local_i], BUNDLE.feature_cols)
        is_hard = row["tamper_flag"] == 1
        results.append(Alert(
            alert_id=f"ALT-{row['customer_id']}-{row['date'].strftime('%Y%m%d')}",
            customer_id=row["customer_id"], cluster_id=row["cluster_id"], timestamp=row["date"].date(),
            severity="hard" if is_hard else "model",
            confidence=1.0 if is_hard else round(float(proba[i]), 3),
            explanation=factors_to_sentence(top_factors), top_factors=top_factors,
        ))
    return sorted(results, key=lambda a: -a.confidence)[:200]


class AlertConfirmation(BaseModel):
    customer_id: str
    incident_date: date
    incident_type: str  # bypass, under_reporting, neutral_interference, magnetic_interference, partial_bypass, reverse_current, other
    confirmed_by: Optional[str] = "dashboard_review"


@app.post("/api/alerts/confirm")
def confirm_alert(body: AlertConfirmation, enforced_scope: Optional[str] = Depends(require_api_key)):
    """
    Turns a model-generated alert into a real, permanent ground-truth record —
    this is THE feedback loop that lets NairaMeter's model keep improving.
    Every confirmation here becomes a new labeled example train_model.py can
    learn from next time it's retrained (via get_confirmed_incidents() in
    db.py / the incidents table directly).
    """
    valid_types = {"bypass", "under_reporting", "neutral_interference",
                   "magnetic_interference", "partial_bypass", "reverse_current", "other"}
    if body.incident_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"incident_type must be one of {sorted(valid_types)}")

    if enforced_scope:
        owner = query_df("SELECT source_partner FROM offtakers WHERE offtaker_id = %s", (body.customer_id,))
        if owner.empty or owner.iloc[0]["source_partner"] != enforced_scope:
            raise HTTPException(status_code=404, detail=f"Customer '{body.customer_id}' not found")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO incidents (meter_id, incident_date, incident_type, confirmed, confirmed_by, confirmed_at)
                VALUES (%s, %s, %s, true, %s, now())
                ON CONFLICT (meter_id, incident_date) DO UPDATE SET
                    incident_type = EXCLUDED.incident_type,
                    confirmed = true,
                    confirmed_by = EXCLUDED.confirmed_by,
                    confirmed_at = now();
            """, (body.customer_id, body.incident_date, body.incident_type, body.confirmed_by))
        conn.commit()
    return {"customer_id": body.customer_id, "incident_date": body.incident_date, "status": "confirmed"}


@app.get("/api/whoami")
def whoami(enforced_scope: Optional[str] = Depends(require_api_key)):
    """
    Lets the dashboard know whether the current key is full-access or locked
    to one partner, so it can hide the data-source selector entirely for a
    restricted key rather than showing a choice that isn't really available.
    """
    return {"scope": enforced_scope}


@app.get("/api/health")
def health():
    try:
        n = query_df("SELECT COUNT(*) AS n FROM meter_readings").iloc[0]["n"]
        return {"status": "ok", "readings_in_db": int(n), "model_loaded": BUNDLE.loaded}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
