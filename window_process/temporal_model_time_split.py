#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/IDS2017/cleaned/windowed_combined.csv"
RESULTS_DIR = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/results"

RESULTS_TXT = os.path.join(RESULTS_DIR, "temporal_results_time_split.txt")
RESULTS_JSON = os.path.join(RESULTS_DIR, "temporal_results_time_split.json")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        c = str(col).strip().lower()
        if c in {"unnamed: 0", "index", ""}:
            drop_cols.append(col)

    if drop_cols:
        print(f"[INFO] Dropping useless columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required = ["target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "target",
        "window_start",
        "window_end",
        "window_size",
        "window_attack_count",
        "ID",
        "Label",
        "Attack_Type",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_model_df(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    model_df = df[feature_cols + ["target"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df[col] = model_df[col].replace([np.inf, -np.inf], np.nan)

    model_df["target"] = pd.to_numeric(model_df["target"], errors="coerce")
    model_df = model_df.dropna(subset=["target"])
    model_df["target"] = model_df["target"].astype(int)

    return model_df


def time_split(X: pd.DataFrame, y: pd.Series):
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_val = X.iloc[train_end:val_end].copy()
    y_val = y.iloc[train_end:val_end].copy()

    X_test = X.iloc[val_end:].copy()
    y_test = y.iloc[val_end:].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def fill_missing_with_train_medians(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
):
    medians = {}

    for col in feature_cols:
        median_value = X_train[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        medians[col] = median_value

    for col in feature_cols:
        X_train[col] = X_train[col].fillna(medians[col])
        X_val[col] = X_val[col].fillna(medians[col])
        X_test[col] = X_test[col].fillna(medians[col])

    return X_train, X_val, X_test, medians


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


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

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[INFO] Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    print(f"[INFO] Loaded shape: {df.shape}")
    print(f"[INFO] Number of columns: {len(df.columns)}")

    validate_required_columns(df)

    # Keep chronological order. If window_start exists, sort by it.
    if "window_start" in df.columns:
        df = df.sort_values(by="window_start").reset_index(drop=True)
        print("[INFO] Sorted by window_start for time-aware splitting.")
    else:
        df = df.reset_index(drop=True)
        print("[WARNING] window_start not found. Using current row order as time order.")

    print("\n[INFO] Target counts before cleaning:")
    print(df["target"].value_counts(dropna=False))

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns available for training.")

    print(f"\n[INFO] Number of feature columns: {len(feature_cols)}")
    print("[INFO] First 20 feature columns:")
    print(feature_cols[:20])

    model_df = prepare_model_df(df, feature_cols)

    X = model_df[feature_cols].copy()
    y = model_df["target"].copy()

    unique_labels = sorted(y.dropna().unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError(f"Only one class found in target: {unique_labels}")

    print(f"\n[INFO] Final modeling shape: {model_df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = time_split(X, y)

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Val shape  : {X_val.shape}")
    print(f"[INFO] Test shape : {X_test.shape}")

    print("\n[INFO] Train class distribution:")
    print(y_train.value_counts())

    print("\n[INFO] Validation class distribution:")
    print(y_val.value_counts())

    print("\n[INFO] Test class distribution:")
    print(y_test.value_counts())

    X_train, X_val, X_test, medians = fill_missing_with_train_medians(
        X_train, X_val, X_test, feature_cols
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    }

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    best_model = None
    best_params = None
    best_val_f1 = -1.0
    all_trials = []

    print("\n[INFO] Starting validation-based model selection...")

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_metrics = evaluate(y_val, val_pred)

        trial = {
            "params": params,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        all_trials.append(trial)

        print(
            f"[INFO] Params={params} | "
            f"Val Acc={val_metrics['accuracy']:.6f} | "
            f"Val Prec={val_metrics['precision']:.6f} | "
            f"Val Rec={val_metrics['recall']:.6f} | "
            f"Val F1={val_metrics['f1']:.6f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_params = params
            best_model = model

    print("\n[INFO] Best validation parameters:")
    print(best_params)
    print(f"[INFO] Best validation F1: {best_val_f1:.6f}")

    print("\n[INFO] Evaluating best model on test set...")
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)

    print("\n=== TIME-AWARE TEMPORAL MODEL RESULTS ===")
    print(f"Best params: {best_params}")
    print(f"Validation F1: {best_val_f1:.6f}")
    print(f"Test Accuracy : {test_metrics['accuracy']:.6f}")
    print(f"Test Precision: {test_metrics['precision']:.6f}")
    print(f"Test Recall   : {test_metrics['recall']:.6f}")
    print(f"Test F1       : {test_metrics['f1']:.6f}")
    print("\nTest confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nTest classification report:")
    print(test_metrics["classification_report"])

    feature_importances = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_features = feature_importances.head(30).to_dict(orient="records")

    results = {
        "input_file": INPUT_FILE,
        "n_rows_total": int(len(model_df)),
        "n_features": int(len(feature_cols)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "best_params": best_params,
        "best_validation_f1": float(best_val_f1),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "feature_columns": feature_cols,
        "top_feature_importances": top_features,
        "all_validation_trials": all_trials,
    }

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("=== TIME-AWARE TEMPORAL MODEL RESULTS ===\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows total: {len(model_df)}\n")
        f.write(f"Number of features: {len(feature_cols)}\n")
        f.write(f"Train rows: {len(X_train)}\n")
        f.write(f"Validation rows: {len(X_val)}\n")
        f.write(f"Test rows: {len(X_test)}\n\n")

        f.write(f"Best params: {best_params}\n")
        f.write(f"Best validation F1: {best_val_f1:.6f}\n\n")

        f.write(f"Test Accuracy : {test_metrics['accuracy']:.6f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.6f}\n")
        f.write(f"Test Recall   : {test_metrics['recall']:.6f}\n")
        f.write(f"Test F1       : {test_metrics['f1']:.6f}\n\n")

        f.write("Test confusion matrix:\n")
        f.write(str(np.array(test_metrics["confusion_matrix"])))
        f.write("\n\nTest classification report:\n")
        f.write(test_metrics["classification_report"])
        f.write("\n\nTop 30 feature importances:\n")
        f.write(feature_importances.head(30).to_string(index=False))

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Saved text results to: {RESULTS_TXT}")
    print(f"[INFO] Saved JSON results to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()