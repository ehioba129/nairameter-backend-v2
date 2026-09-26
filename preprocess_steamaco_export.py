"""
SteamaCo/CESEL Export Preprocessor
====================================
SteamaCo's Nimbus platform exports one CSV per parameter, per meter (13
separate files), each with a 4-line metadata header. This is fundamentally
different from the single-combined-file format ingestion_service.py expects
from other partners — this script merges a meter's separate files into one
clean, wide-format CSV first, which can then go through the normal ingestion
pipeline.

Usage:
    python preprocess_steamaco_export.py --folder "path/to/extracted/zip" --out cesel_meter_combined.csv

Expects files named exactly as SteamaCo exports them:
    "Meter - <meter_id> - <Parameter Name>.csv"

IMPORTANT — two things requiring your confirmation before this data is used
for real detection results, not just a pipeline test:

1. "Import Energy" vs "Electricity" (labeled "Energy Use (kWh)"): these are
   genuinely different values in the export — Energy Use runs consistently
   30-50% higher than Import Energy for the same days. This script uses
   Import Energy as kwh_recorded, since it's the standard utility metering
   term for grid-imported (billable) energy. If CESEL's mini-grid also
   includes local solar generation, "Energy Use" might represent total
   consumption including self-generated power, while "Import Energy" is
   only the grid-supplied portion. CONFIRM WITH CESEL which one they
   actually bill customers on before treating results as meaningful.

2. cluster_id and customer_type are NOT present in SteamaCo's export at
   all. Every customer from this source will default to "UNASSIGNED" /
   "residential" unless you supply this separately (see --cluster and
   --customer-type arguments below, or edit the output CSV directly).
"""

import argparse
import glob
import os
import re

import pandas as pd

# Map each SteamaCo export filename suffix to (internal_column_name, unit_conversion)
# unit_conversion is a function applied to the raw value, or None if no conversion needed.
PARAM_FILES = {
    "Phase 1 Line to Neutral Voltage": ("voltage", None),
    "Current": ("current", None),
    "Active Power": ("active_power_kw", lambda w: w / 1000.0),   # W -> kW
    "Reactive Power": ("reactive_power_kvar", lambda var: var / 1000.0),  # var -> kvar
    "Power Factor": ("power_factor", None),
    "Import Energy": ("kwh_recorded", None),  # see docstring — confirm with CESEL
}
# Captured but not part of the core internal schema yet — kept in the output
# for reference / future use, not dropped.
EXTRA_PARAM_FILES = {
    "Apparent Power": "apparent_power_va",
    "Frequency": "frequency_hz",
    "Electricity": "energy_use_kwh",  # the "Energy Use (kWh)" alternative — see docstring
    "Neutral Current": "neutral_current",
    "Reactive Energy": "reactive_energy",
    "Power Uptime": "power_uptime_pct",
    "Overall Uptime": "overall_uptime_pct",
}


def load_steamaco_param_file(filepath: str, value_col_name: str, convert=None) -> pd.DataFrame:
    """Loads one SteamaCo per-parameter file, skipping its 4-line metadata header."""
    df = pd.read_csv(filepath, skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    if df.empty or len(df.columns) < 2:
        # Some parameters (e.g., Reactive Power, Uptime) can be entirely empty for a
        # given meter — return an empty frame with the right shape rather than erroring.
        return pd.DataFrame(columns=["reading_time", value_col_name])
    ts_col, val_col = df.columns[0], df.columns[1]
    df = df.rename(columns={ts_col: "reading_time", val_col: value_col_name})
    df["reading_time"] = pd.to_datetime(df["reading_time"].astype(str).str.strip(), errors="coerce")
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
    df = df.dropna(subset=["reading_time"])
    if convert:
        df[value_col_name] = df[value_col_name].apply(convert)
    return df[["reading_time", value_col_name]]


def find_meter_id(folder: str) -> str:
    """Extracts the meter ID from any file in the folder, e.g. 'Meter - 0179002097105 - Current.csv'."""
    files = glob.glob(os.path.join(folder, "Meter - * - *.csv"))
    if not files:
        raise FileNotFoundError(f"No SteamaCo-format files found in {folder}")
    match = re.search(r"Meter - (.+?) - ", os.path.basename(files[0]))
    if not match:
        raise ValueError(f"Could not parse meter ID from filename: {files[0]}")
    return match.group(1)


def process_meter_folder(folder: str, cluster_id: str = "UNASSIGNED", customer_type: str = "residential") -> pd.DataFrame:
    """Merges every parameter file for one meter into a single wide-format DataFrame."""
    meter_id = find_meter_id(folder)
    print(f"Processing meter: {meter_id}")

    merged = None
    for suffix, (col_name, convert) in PARAM_FILES.items():
        filepath = os.path.join(folder, f"Meter - {meter_id} - {suffix}.csv")
        if not os.path.exists(filepath):
            print(f"  WARNING: expected file not found, skipping: {suffix}")
            continue
        param_df = load_steamaco_param_file(filepath, col_name, convert)
        print(f"  {suffix}: {len(param_df)} data rows")
        merged = param_df if merged is None else merged.merge(param_df, on="reading_time", how="outer")

    # Extra/bonus fields — merged in too, kept for reference even though not
    # used by the current model's feature set.
    for suffix, col_name in EXTRA_PARAM_FILES.items():
        filepath = os.path.join(folder, f"Meter - {meter_id} - {suffix}.csv")
        if not os.path.exists(filepath):
            continue
        param_df = load_steamaco_param_file(filepath, col_name)
        merged = merged.merge(param_df, on="reading_time", how="outer")

    merged = merged.sort_values("reading_time").reset_index(drop=True)
    merged.insert(0, "customer_id", meter_id)
    merged.insert(1, "cluster_id", cluster_id)
    merged.insert(2, "customer_type", customer_type)

    # Fields SteamaCo doesn't provide at all — explicit, not silently dropped.
    merged["tamper_flag"] = 0    # not available from this data source
    merged["is_outage"] = 0      # not available from this data source (uptime fields were empty)

    merged = merged.rename(columns={"reading_time": "date"})
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing one meter's extracted SteamaCo CSV files")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--cluster", default="UNASSIGNED", help="Cluster ID for this customer (SteamaCo export doesn't include this)")
    parser.add_argument("--customer-type", default="residential", choices=["residential", "commercial", "industrial"])
    args = parser.parse_args()

    result = process_meter_folder(args.folder, cluster_id=args.cluster, customer_type=args.customer_type)
    result.to_csv(args.out, index=False)
    print(f"\nSaved {len(result)} rows to {args.out}")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")

    core_fields = ["voltage", "current", "active_power_kw", "power_factor", "kwh_recorded"]
    present_fields = [f for f in core_fields if f in result.columns]
    absent_fields = [f for f in core_fields if f not in result.columns]

    if absent_fields:
        print(f"\nWARNING: these core fields have NO data at all for this meter "
              f"(the source file was missing entirely, not just some days): {absent_fields}")
    if present_fields:
        missing = result[present_fields].isna().sum()
        print(f"\nMissing values per core field that IS present:\n{missing}")


if __name__ == "__main__":
    main()
