#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/combined.csv"
OUTPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/windowed_grouped_combined.csv"

WINDOW_SIZE = 20
STEP_SIZE = 10


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_selected_features() -> dict:
    return {
        "mean_std": [
            "Init Fwd Win Bytes",
            "Down/Up Ratio",
            "Packet Length Variance",
            "Fwd Packet Length Std",
            "Packet Length Min",
            "Init Bwd Win Bytes",
            "Avg Packet Size",
            "Flow IAT Mean",
        ],
        "sum": [
            "FIN Flag Count",
        ],
        "mean_only": [
            "Protocol",
        ],
    }


def validate_columns(df: pd.DataFrame, feature_groups: dict) -> None:
    required = ["Label", "Attack_Type"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")

    selected = []
    for group in feature_groups.values():
        selected.extend(group)

    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")


def normalize_attack_type(x) -> str:
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "Unknown"
    return s


def make_binary_label_from_group(group_name: str) -> int:
    return 0 if str(group_name).strip().lower() == "benign" else 1


def main():
    print(f"[INFO] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = clean_column_names(df)

    drop_cols = [c for c in df.columns if c.strip().lower() in {"unnamed: 0", "index", ""}]
    if drop_cols:
        print(f"[INFO] Dropping useless columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[INFO] Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    feature_groups = get_selected_features()
    validate_columns(df, feature_groups)

    selected_features = []
    for group in feature_groups.values():
        selected_features.extend(group)

    keep_cols = ["Label", "Attack_Type"] + selected_features
    if "ID" in df.columns:
        keep_cols = ["ID"] + keep_cols

    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    df = df[keep_cols].copy()

    # Clean label
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    # Normalize attack type
    df["Attack_Type"] = df["Attack_Type"].apply(normalize_attack_type)

    # Clean numeric features
    for col in selected_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    print(f"[INFO] Input rows: {len(df)}")
    print(f"[INFO] Window size: {WINDOW_SIZE}")
    print(f"[INFO] Step size: {STEP_SIZE}")
    print("[INFO] Selected features:")
    print(feature_groups)

    print("\n[INFO] Attack_Type counts:")
    print(df["Attack_Type"].value_counts(dropna=False))

    window_rows = []
    total_windows = 0

    # Group by attack type so windows are behaviorally coherent
    for attack_type, gdf in df.groupby("Attack_Type", sort=False):
        gdf = gdf.reset_index(drop=True)

        if len(gdf) < WINDOW_SIZE:
            print(f"[INFO] Skipping group '{attack_type}' because size {len(gdf)} < window size")
            continue

        group_binary_label = make_binary_label_from_group(attack_type)

        print(f"[INFO] Processing group '{attack_type}' with {len(gdf)} rows")

        for start in range(0, len(gdf) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            w = gdf.iloc[start:end]

            row = {
                "group_attack_type": attack_type,
                "window_start_in_group": start,
                "window_end_in_group": end - 1,
                "window_size": WINDOW_SIZE,
                "Label": group_binary_label,
            }

            # mean + std features
            for col in feature_groups["mean_std"]:
                vals = w[col].values
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals))

            # sum features
            for col in feature_groups["sum"]:
                vals = w[col].values
                row[f"{col}_sum"] = float(np.sum(vals))

            # mean-only features
            for col in feature_groups["mean_only"]:
                vals = w[col].values
                row[f"{col}_mean"] = float(np.mean(vals))

            window_rows.append(row)
            total_windows += 1

            if total_windows % 20000 == 0:
                print(f"[INFO] Processed windows: {total_windows}")

    window_df = pd.DataFrame(window_rows)

    print(f"\n[INFO] Final windowed shape: {window_df.shape}")
    print("\n[INFO] Window Label counts:")
    print(window_df["Label"].value_counts(dropna=False))

    print("\n[INFO] Group attack type counts:")
    print(window_df["group_attack_type"].value_counts(dropna=False))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    window_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[INFO] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()