#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np


INPUT_FOLDER = "/Users/farihatahosinboishakhi/Documents/Uconn Courses/Spring26/CSE5300/Safegaurd/data/archive/"
OUTPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn Courses/Spring26/CSE5300/Safegaurd/data/clean_all_flows.csv"


def normalize_column_names(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_existing(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.parquet"))

dfs = []

print(f"[INFO] Found {len(all_files)} parquet files")

for file in all_files:
    print(f"[INFO] Reading {file}")

    df = pd.read_parquet(file)
    df = normalize_column_names(df)

    timestamp_candidates = ["Timestamp", "timestamp"]
    label_candidates = ["Label", "label"]
    src_candidates = ["Src IP", "Source IP", "src_ip"]
    dst_candidates = ["Dst IP", "Destination IP", "dst_ip"]

    timestamp_col = find_first_existing(timestamp_candidates, df.columns)
    label_col = find_first_existing(label_candidates, df.columns)
    src_col = find_first_existing(src_candidates, df.columns)
    dst_col = find_first_existing(dst_candidates, df.columns)

    if timestamp_col is None or label_col is None:
        print(f"[WARN] skipping {file}")
        continue

    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    df[label_col] = df[label_col].astype(str).str.strip()

    def make_binary(x):
        if x.lower() in ["benign", "normal"]:
            return 0
        return 1

    df["target"] = df[label_col].apply(make_binary)

    selected_features = [
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Pkt Len Mean",
        "SYN Flag Cnt",
        "ACK Flag Cnt"
    ]

    available = [c for c in selected_features if c in df.columns]

    keep_cols = [timestamp_col, label_col, "target"]

    if src_col:
        keep_cols.append(src_col)

    if dst_col:
        keep_cols.append(dst_col)

    keep_cols.extend(available)

    df = df[keep_cols]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[timestamp_col])

    dfs.append(df)


print("[INFO] Concatenating all files...")
full_df = pd.concat(dfs, ignore_index=True)

print("[INFO] Sorting by timestamp...")
timestamp_col = find_first_existing(["Timestamp", "timestamp"], full_df.columns)

full_df = full_df.sort_values(timestamp_col).reset_index(drop=True)

print("[INFO] Final shape:", full_df.shape)

full_df.to_csv(OUTPUT_FILE, index=False)

print("[INFO] Saved:", OUTPUT_FILE)