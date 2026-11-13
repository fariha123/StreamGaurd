#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/combined.csv"
OUTPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/windowed_combined.csv"

WINDOW_SIZE = 20
STEP_SIZE = 10


def make_binary_target(label_value) -> int:
    s = str(label_value).strip().lower()
    if s == "benign":
        return 0
    return 1


def main():
    print(f"[INFO] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop accidental index columns if present
    drop_cols = [c for c in df.columns if c.strip().lower() in {"unnamed: 0", "index", ""}]
    if drop_cols:
        print(f"[INFO] Dropping useless columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Remove duplicate columns if any
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[INFO] Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    if "Label" not in df.columns:
        raise ValueError("Column 'Label' not found.")

    # Create binary row target only for window labeling
    #df["target"] = df["Label"].apply(make_binary_target)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Explicitly exclude non-feature columns from aggregation
    exclude = {"ID", "Label", "Attack_Type"}

    feature_cols = [c for c in numeric_cols if c not in exclude]

    print(f"[INFO] Number of numeric feature columns used for aggregation: {len(feature_cols)}")
    print("[INFO] First 15 aggregation features:")
    print(feature_cols[:15])

    # Clean numeric features
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    print(f"[INFO] Rows in input: {len(df)}")

    window_rows = []
    total_windows = 0

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        w = df.iloc[start:end]

        row = {
            "window_start": start,
            "window_end": end - 1,
            "window_size": WINDOW_SIZE,
            "window_attack_count": int(w["target"].sum()),
            "target": int((w["target"] == 1).any()),
        }

        # Aggregate only true numeric flow features
        for col in feature_cols:
            vals = w[col].values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
            row[f"{col}_min"] = float(np.min(vals))
            row[f"{col}_max"] = float(np.max(vals))

        window_rows.append(row)
        total_windows += 1

        if total_windows % 20000 == 0:
            print(f"[INFO] Processed windows: {total_windows}")

    window_df = pd.DataFrame(window_rows)

    print(f"[INFO] Final windowed shape: {window_df.shape}")
    print("[INFO] Window target distribution:")
    print(window_df["target"].value_counts(dropna=False))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    window_df.to_csv(OUTPUT_FILE, index=False)

    print(f"[INFO] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()