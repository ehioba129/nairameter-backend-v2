"""
NairaMeter Ingestion Service
=============================
Takes a raw CSV export from a data-collaboration partner (Husk Power, CESEL,
etc.), validates it, transforms it into NairaMeter's internal schema, and
loads it into the meter_readings table (schema.sql).

This is the "Ingestion Service" box in Section 2 of the MVP Technical
Specification's architecture diagram — the component sitting between raw
partner data and the database.

Design notes:
  - Partner CSVs won't all use the same column names or units. PARTNER_COLUMN_MAPS
    below defines a translation per partner; add a new entry when onboarding
    a new data partner rather than writing a one-off script each time.
  - Validation happens BEFORE loading — bad rows are rejected and logged, not
    silently inserted. Every run is recorded in ingestion_runs for auditability.
  - Loading uses "upsert" semantics (ON CONFLICT DO UPDATE) so re-running the
    same file twice is safe and won't create duplicate rows.

Run:
    pip install psycopg2-binary pandas
    python ingestion_service.py --file husk_export.csv --partner husk_power \
        --db-url postgresql://user:pass@host:5432/nairameter
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


# =====================================================================
# 1. PARTNER COLUMN MAPPING
# =====================================================================
# Maps each partner's raw CSV column names to NairaMeter's internal schema.
# Add a new entry here whenever a new data partner is onboarded — no other
# code needs to change.
INTERNAL_COLUMNS = [
    "meter_id", "reading_time", "voltage", "current", "active_power_kw",
    "reactive_power_kvar", "power_factor", "kwh_recorded", "tamper_status", "is_outage",
]

PARTNER_COLUMN_MAPS = {
    "husk_power": {
        "customer_id": "meter_id",
        "date": "reading_time",
        "voltage": "voltage",
        "current": "current",
        "active_power_kw": "active_power_kw",
        "reactive_power_kvar": "reactive_power_kvar",
        "power_factor": "power_factor",
        "kwh_recorded": "kwh_recorded",
        "tamper_flag": "tamper_status",
        "is_outage": "is_outage",
    },
    "cesel": {
        # Illustrative — update once CESEL's actual export format is confirmed.
        # This is exactly why the mapping is config, not hardcoded logic.
        "customer_id": "meter_id",
        "date": "reading_time",
        "voltage": "voltage",
        "current": "current",
        "active_power_kw": "active_power_kw",
        "reactive_power_kvar": "reactive_power_kvar",
        "power_factor": "power_factor",
        "kwh_recorded": "kwh_recorded",
        "tamper_flag": "tamper_status",
        "is_outage": "is_outage",
    },
    # Matches the schema of meter_readings_daily.csv used for MVP development/testing.
    "internal_test_schema": {
        "customer_id": "meter_id",
        "date": "reading_time",
        "voltage": "voltage",
        "current": "current",
        "active_power_kw": "active_power_kw",
        "reactive_power_kvar": "reactive_power_kvar",
        "power_factor": "power_factor",
        "kwh_recorded": "kwh_recorded",
        "tamper_flag": "tamper_status",
        "is_outage": "is_outage",
    },
}


# =====================================================================
# 2. VALIDATION RULES
# =====================================================================
# Range checks mirror the CHECK constraints in schema.sql — validating here
# lets us reject and *explain* bad rows before they ever hit the database,
# rather than getting an opaque constraint-violation error from Postgres.
VALIDATION_RULES = {
    "voltage": (0, 500),
    # Current can legitimately be negative — a reversed current reading is itself
    # a theft signature ("reverse_current" tampering), not a data error. Reject
    # only implausible magnitudes, not the sign.
    "current": (-100, 100),
    "power_factor": (-1, 1),
    "kwh_recorded": (0, None),
}


@dataclass
class ValidationResult:
    valid_df: pd.DataFrame
    rejected_rows: list = field(default_factory=list)  # list of (row_index, reason)

    @property
    def n_valid(self):
        return len(self.valid_df)

    @property
    def n_rejected(self):
        return len(self.rejected_rows)


def validate_schema(df: pd.DataFrame, column_map: dict) -> None:
    """Raises a clear error if required raw columns are missing — fail fast,
    before any row-level processing starts."""
    missing = [col for col in column_map if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing expected columns: {missing}. "
            f"Check the partner's export format against PARTNER_COLUMN_MAPS."
        )


def validate_rows(df: pd.DataFrame) -> ValidationResult:
    """
    Row-level data quality checks, applied after column mapping/renaming.
    Returns the subset of rows that pass, plus a log of what was rejected
    and why — this log is what gets summarized into ingestion_runs.
    """
    rejected = []
    keep_mask = pd.Series(True, index=df.index)

    # Required fields must not be null
    for col in ["meter_id", "reading_time"]:
        bad = df[col].isna()
        for idx in df.index[bad]:
            rejected.append((idx, f"Missing required field: {col}"))
        keep_mask &= ~bad

    # Range checks
    for col, (lo, hi) in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        for idx in df.index[bad & keep_mask]:
            rejected.append((idx, f"{col}={df.loc[idx, col]} outside valid range [{lo}, {hi}]"))
        keep_mask &= ~bad

    # Duplicate (meter_id, reading_time) pairs within this file — keep first, reject rest
    dupe_mask = df.duplicated(subset=["meter_id", "reading_time"], keep="first")
    for idx in df.index[dupe_mask & keep_mask]:
        rejected.append((idx, "Duplicate meter_id + reading_time within this file"))
    keep_mask &= ~dupe_mask

    return ValidationResult(valid_df=df[keep_mask].copy(), rejected_rows=rejected)


# =====================================================================
# 3. TRANSFORMATION
# =====================================================================
def transform(raw_df: pd.DataFrame, partner: str) -> pd.DataFrame:
    """
    Renames partner-specific columns to NairaMeter's internal schema and
    coerces types. Returns a dataframe with exactly INTERNAL_COLUMNS.
    """
    if partner not in PARTNER_COLUMN_MAPS:
        raise ValueError(
            f"Unknown partner '{partner}'. Add a column mapping to "
            f"PARTNER_COLUMN_MAPS before ingesting their data."
        )
    column_map = PARTNER_COLUMN_MAPS[partner]
    validate_schema(raw_df, column_map)

    df = raw_df.rename(columns=column_map)[list(set(column_map.values()))].copy()

    # Type coercion
    df["reading_time"] = pd.to_datetime(df["reading_time"], errors="coerce")
    numeric_cols = ["voltage", "current", "active_power_kw", "reactive_power_kvar",
                     "power_factor", "kwh_recorded"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["tamper_status", "is_outage"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Ensure every internal column exists, even if the partner doesn't supply it
    for col in INTERNAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[INTERNAL_COLUMNS]


# =====================================================================
# 4. DATABASE LOADING
# =====================================================================
def load_to_database(df: pd.DataFrame, partner: str, run_id: int, conn) -> int:
    """
    Bulk-upserts validated rows into meter_readings using psycopg2's
    execute_values for efficient batch insertion. ON CONFLICT DO UPDATE
    makes re-running the same file idempotent — no duplicate rows, and a
    re-ingested file simply refreshes the existing readings.

    Requires: psycopg2. Not exercised in this environment (no live Postgres
    instance available here) — the validation/transformation logic above
    IS fully tested against real data; this function follows standard,
    well-established psycopg2 patterns.
    """
    from psycopg2.extras import execute_values
    import math

    def nan_to_none(val):
        """Postgres treats floating-point NaN as a real value that fails every
        numeric comparison — including CHECK constraints meant to allow missing
        data. A true NULL passes those constraints; a NaN does not. Any pandas
        NaN must become a Python None before it reaches the database, or rows
        with a gapped reading get rejected as if they were out-of-range data."""
        if val is None:
            return None
        try:
            if isinstance(val, float) and math.isnan(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    records = [
        (
            row.meter_id, row.reading_time, nan_to_none(row.voltage), nan_to_none(row.current),
            nan_to_none(row.active_power_kw), nan_to_none(row.reactive_power_kvar), nan_to_none(row.power_factor),
            nan_to_none(row.kwh_recorded), row.tamper_status, row.is_outage, partner, run_id,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO meter_readings (
            meter_id, reading_time, voltage, current, active_power_kw,
            reactive_power_kvar, power_factor, kwh_recorded, tamper_status,
            is_outage, source_partner, ingestion_run_id
        )
        VALUES %s
        ON CONFLICT (meter_id, reading_time) DO UPDATE SET
            voltage = EXCLUDED.voltage,
            current = EXCLUDED.current,
            active_power_kw = EXCLUDED.active_power_kw,
            reactive_power_kvar = EXCLUDED.reactive_power_kvar,
            power_factor = EXCLUDED.power_factor,
            kwh_recorded = EXCLUDED.kwh_recorded,
            tamper_status = EXCLUDED.tamper_status,
            is_outage = EXCLUDED.is_outage,
            ingestion_run_id = EXCLUDED.ingestion_run_id;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    conn.commit()
    return len(records)


def ensure_offtakers_exist(meter_metadata: list, cluster_id_default: str, customer_type_default: str,
                            partner: str, conn) -> None:
    """
    Ensures every meter_id in this batch has a corresponding offtakers row
    (foreign key requirement).

    meter_metadata is a list of (meter_id, cluster_id, customer_type) tuples.
    Where a real cluster_id/customer_type was supplied in the source data,
    that real value is used — the defaults are only a fallback for meters
    where the partner's file genuinely didn't include this information.

    This replaces the earlier version, which applied a single hardcoded
    default to every meter in the batch regardless of what the source file
    actually contained — that bug is what caused every customer to show
    "UNASSIGNED" in the dashboard even when real cluster data existed.
    """
    from psycopg2.extras import execute_values

    records = [
        (mid, (cid if cid and str(cid).strip() else cluster_id_default),
         (ctype if ctype and str(ctype).strip() else customer_type_default), partner)
        for mid, cid, ctype in meter_metadata
    ]
    sql = """
        INSERT INTO offtakers (offtaker_id, cluster_id, customer_type, source_partner)
        VALUES %s
        ON CONFLICT (offtaker_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    conn.commit()


# =====================================================================
# 5. MAIN PIPELINE
# =====================================================================
def run_ingestion(file_path: str, partner: str, db_url: str = None) -> dict:
    """
    End-to-end ingestion: load CSV -> validate schema -> transform ->
    validate rows -> load to DB -> log the run. Returns a summary dict.
    """
    print(f"[{datetime.now().isoformat()}] Starting ingestion: {file_path} (partner={partner})")

    # dtype=str on customer_id prevents pandas from re-inferring this all-digit
    # column as an integer and silently dropping a leading zero (e.g. CESEL's
    # meter IDs like "0179002097105" becoming "179002097105") — this exact bug
    # already happened once when combining files; this read needs the same
    # protection independently, since it's a separate pd.read_csv call.
    raw_df = pd.read_csv(file_path, dtype={"customer_id": str})
    rows_received = len(raw_df)
    print(f"  Loaded {rows_received} raw rows")

    # Capture real per-meter cluster/customer-type from the source file BEFORE
    # transform() drops them (transform only keeps meter_readings-table columns).
    # Falls back gracefully if the partner's file doesn't include these at all.
    meter_metadata = []
    if "customer_id" in raw_df.columns:
        meta_cols = ["customer_id"]
        meta_cols += [c for c in ["cluster_id", "customer_type"] if c in raw_df.columns]
        meta_df = raw_df[meta_cols].drop_duplicates(subset=["customer_id"])
        for _, row in meta_df.iterrows():
            meter_metadata.append((
                row["customer_id"],
                row.get("cluster_id"),
                row.get("customer_type"),
            ))

    transformed_df = transform(raw_df, partner)
    result = validate_rows(transformed_df)
    print(f"  Validation: {result.n_valid} valid, {result.n_rejected} rejected")

    if result.n_rejected:
        print("  Sample rejection reasons:")
        for idx, reason in result.rejected_rows[:5]:
            print(f"    Row {idx}: {reason}")

    summary = {
        "file_name": file_path,
        "source_partner": partner,
        "rows_received": rows_received,
        "rows_loaded": result.n_valid,
        "rows_rejected": result.n_rejected,
        "status": "success" if result.n_rejected == 0 else ("partial" if result.n_valid > 0 else "failed"),
    }

    if db_url is None:
        print("  [DRY RUN] No --db-url provided — skipping database load.")
        print(f"  Summary: {summary}")
        return summary

    import psycopg2
    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ingestion_runs (source_partner, file_name, rows_received, status)
               VALUES (%s, %s, %s, 'running') RETURNING run_id""",
            (partner, file_path, rows_received),
        )
        run_id = cur.fetchone()[0]
        conn.commit()

        # Fall back to defaults-only if the source file had no customer_id column
        # at all (shouldn't happen, but keeps this robust for any partner format).
        if not meter_metadata:
            meter_metadata = [(mid, None, None) for mid in result.valid_df["meter_id"].unique().tolist()]

        ensure_offtakers_exist(
            meter_metadata,
            cluster_id_default="UNASSIGNED", customer_type_default="residential",
            partner=partner, conn=conn,
        )
        n_loaded = load_to_database(result.valid_df, partner, run_id, conn)

        cur.execute(
            """UPDATE ingestion_runs SET completed_at = now(), rows_loaded = %s,
               rows_rejected = %s, status = %s WHERE run_id = %s""",
            (n_loaded, result.n_rejected, summary["status"], run_id),
        )
        conn.commit()
        print(f"  Loaded {n_loaded} rows into meter_readings (run_id={run_id})")
    finally:
        conn.close()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to partner CSV export")
    parser.add_argument("--partner", required=True, choices=list(PARTNER_COLUMN_MAPS.keys()))
    parser.add_argument("--db-url", default=None, help="postgresql://user:pass@host:port/db (omit for dry run)")
    args = parser.parse_args()

    summary = run_ingestion(args.file, args.partner, args.db_url)
    sys.exit(0 if summary["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
