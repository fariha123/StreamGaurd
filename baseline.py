#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/combined.csv"
RESULTS_DIR = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/results"
RESULTS_TXT = os.path.join(RESULTS_DIR, "baseline_results.txt")
RESULTS_JSON = os.path.join(RESULTS_DIR, "baseline_results.json")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []

    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str in {"unnamed: 0", "index", ""}:
            drop_cols.append(col)

    if drop_cols:
        print(f"[INFO] Dropping useless columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    if "Label" not in df.columns:
        raise ValueError("Required column 'Label' not found in input CSV.")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target/leakage/non-feature columns
    exclude = {"Label", "ID", "target"}

    for col in ["Attack_Type"]:
        if col in numeric_cols:
            exclude.add(col)

    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[INFO] Loading file: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {INPUT_FILE}")
        sys.exit(1)

    df = clean_column_names(df)
    df = drop_useless_columns(df)

    # Remove duplicate columns if any
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[INFO] Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    print(f"[INFO] Loaded shape: {df.shape}")
    print(f"[INFO] Number of columns: {len(df.columns)}")

    validate_required_columns(df)

    print("\n[INFO] Label counts:")
    print(df["Label"].value_counts(dropna=False))
    
    print(df.describe())
    print(df.info())

    # Ensure label is numeric 0/1
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    unique_labels = sorted(df["Label"].dropna().unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError(
            f"Only one class found in Label: {unique_labels}. "
            "A baseline classifier for anomaly detection needs both benign and attack rows."
        )

    print("\n[INFO] Unique labels:")
    print(unique_labels)

    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise ValueError("No numeric feature columns available for training.")

    print(f"\n[INFO] Number of numeric feature columns: {len(feature_cols)}")
    print("[INFO] First 15 feature columns:")
    print(feature_cols[:15])

    model_df = df[feature_cols + ["Label"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    model_df = model_df.dropna(subset=["Label"])

    X = model_df[feature_cols]
    y = model_df["Label"]

    print(f"\n[INFO] Final modeling shape: {model_df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape : {X_test.shape}")

    print("\n[INFO] Train class distribution:")
    print(y_train.value_counts())

    print("\n[INFO] Test class distribution:")
    print(y_test.value_counts())

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\n[INFO] Training Random Forest baseline...")
    model.fit(X_train, y_train)

    print("[INFO] Predicting on test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n=== BASELINE RESULTS ===")
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    results = {
        "input_file": INPUT_FILE,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "feature_columns": feature_cols,
    }

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("=== BASELINE RESULTS ===\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Number of features: {len(feature_cols)}\n")
        f.write(f"Train rows: {len(X_train)}\n")
        f.write(f"Test rows: {len(X_test)}\n\n")
        f.write(f"Accuracy : {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall   : {recall:.6f}\n")
        f.write(f"F1       : {f1:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved text results to: {RESULTS_TXT}")
    print(f"[INFO] Saved JSON results to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()