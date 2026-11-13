#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/combined.csv"

def main():
    print(f"[INFO] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Clean columns
    df.columns = [str(c).strip() for c in df.columns]

    # Drop obvious non-feature columns
    drop_cols = [c for c in ["ID", "Attack_Type", "target"] if c in df.columns]

    if drop_cols:
        print(f"[INFO] Dropping: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Ensure Label numeric
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    # Keep numeric only
    numeric_df = df.select_dtypes(include=[np.number])

    # Replace inf/nan
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Correlation with Label
    corr = numeric_df.corr()["Label"].abs().sort_values(ascending=False)

    # Remove Label itself
    corr = corr.drop("Label")

    print("\n=== TOP 20 FEATURES BY CORRELATION WITH LABEL ===\n")
    print(corr.head(20))

if __name__ == "__main__":
    main()