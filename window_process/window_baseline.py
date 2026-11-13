#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

INPUT_FILE = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/windowed_combined.csv"
RESULTS_DIR = "/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/results"
RESULTS_TXT = os.path.join(RESULTS_DIR, "windowed_results.txt")
RESULTS_JSON = os.path.join(RESULTS_DIR, "windowed_results.json")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[INFO] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "target" not in df.columns:
        raise ValueError("Column 'target' not found.")

    feature_cols = [c for c in df.columns if c not in {"target"}]

    # Keep only numeric features
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())

    y = df["target"].astype(int)

    print("[INFO] Window target counts:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape : {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("[INFO] Training windowed model...")
    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n=== WINDOWED RESULTS ===")
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
    }

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("=== WINDOWED RESULTS ===\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Features: {len(feature_cols)}\n")
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

    print(f"[INFO] Saved: {RESULTS_TXT}")
    print(f"[INFO] Saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()