"""
Heart Disease Risk Prediction Using Classification Models
=========================================================
Author: Dylan
Course: Machine Learning
Institution: Fisk University

Description:
    Compares Logistic Regression, Random Forest, and XGBoost
    classifiers on the UCI Heart Disease dataset. Includes
    preprocessing, hyperparameter tuning, evaluation metrics,
    and feature importance analysis.

Dataset:
    UCI Heart Disease Dataset (via Kaggle / UCI Repository)
    Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    (A clean, widely-used version of the Cleveland Heart Disease data)

Usage:
    python heart_disease_classifier.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 1. Load & Inspect Data ────────────────────────────────────────────────────
print("=" * 65)
print("  Heart Disease Risk Prediction — Classification Study")
print("=" * 65)

# Simulate the UCI Heart Disease dataset (Cleveland subset, 303 patients)
# This mirrors the real dataset's exact distribution so the code works
# identically whether you load from CSV or use this synthetic version.
# To use the real file: df = pd.read_csv("heart.csv")
df = pd.read_csv("heart.csv")
df["target"] = (df["HeartDisease"]).astype(int)
df = df.drop(columns=["HeartDisease"])

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

print(f"\n[1] Dataset Loaded")
print(f"    Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"    Target distribution: No Disease={sum(target==0)} | Disease={sum(target==1)}")
print(f"    Missing values: {df.isnull().sum().sum()}")

# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis")
print(df.describe().round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Exploratory Data Analysis — Heart Disease Dataset", fontsize=15, fontweight="bold")

# Age distribution by class
for label, color in [(0, "#2196F3"), (1, "#F44336")]:
    axes[0, 0].hist(df[df.target == label]["age"], bins=20,
                    alpha=0.7, color=color, label=["No Disease", "Disease"][label])
axes[0, 0].set_title("Age Distribution by Class")
axes[0, 0].set_xlabel("Age")
axes[0, 0].legend()

# Chest pain type
cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_counts.plot(kind="bar", ax=axes[0, 1], color=["#2196F3", "#F44336"], alpha=0.8)
axes[0, 1].set_title("Chest Pain Type vs Target")
axes[0, 1].set_xlabel("CP Type (0=Typical Angina)")
axes[0, 1].legend(["No Disease", "Disease"])
axes[0, 1].tick_params(axis='x', rotation=0)

# Max Heart Rate
for label, color in [(0, "#2196F3"), (1, "#F44336")]:
    axes[0, 2].hist(df[df.target == label]["thalach"], bins=20,
                    alpha=0.7, color=color, label=["No Disease", "Disease"][label])
axes[0, 2].set_title("Max Heart Rate by Class")
axes[0, 2].set_xlabel("Max Heart Rate (bpm)")
axes[0, 2].legend()

# Cholesterol boxplot
df.boxplot(column="chol", by="target", ax=axes[1, 0],
           boxprops=dict(color="#333"), medianprops=dict(color="#F44336", linewidth=2))
axes[1, 0].set_title("Cholesterol by Class")
axes[1, 0].set_xlabel("Target (0=No Disease, 1=Disease)")
plt.sca(axes[1, 0])
plt.title("Cholesterol by Class")

# Correlation heatmap
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[1, 1], cmap="RdBu_r",
            center=0, annot=False, fmt=".1f", linewidths=0.4)
axes[1, 1].set_title("Feature Correlation Heatmap")

# Target class balance
counts = df["target"].value_counts()
axes[1, 2].pie(counts, labels=["No Disease", "Disease"],
               colors=["#2196F3", "#F44336"], autopct="%1.1f%%",
               startangle=90, wedgeprops=dict(width=0.6))
axes[1, 2].set_title("Class Distribution")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    → Saved eda_plots.png")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
print("\n[3] Preprocessing")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"    Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Model Training & Evaluation ───────────────────────────────────────────
print("\n[4] Model Training & Evaluation")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=SEED)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, model in models.items():
    X_tr = X_train_s if name in ["Logistic Regression", "SVM"] else X_train
    X_te = X_test_s  if name in ["Logistic Regression", "SVM"] else X_test
    X_cv = X_train_s if name in ["Logistic Regression", "SVM"] else X_train

    cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    results[name] = {
        "model":    model,
        "X_test":   X_te,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc":      roc_auc_score(y_test, y_proba),
        "f1":       f1_score(y_test, y_pred),
        "precision":precision_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "cv_auc":   cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }

    print(f"\n  ── {name} ──")
    print(f"    Accuracy : {results[name]['accuracy']:.3f}")
    print(f"    AUC-ROC  : {results[name]['auc']:.3f}")
    print(f"    F1 Score : {results[name]['f1']:.3f}")
    print(f"    CV AUC   : {results[name]['cv_auc']:.3f} ± {results[name]['cv_std']:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ── 5. Visualizations ────────────────────────────────────────────────────────
print("\n[5] Generating Visualizations")

colors = {"Logistic Regression": "#1565C0", "Random Forest": "#2E7D32", "SVM": "#AD1457"}

# ─ ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax.plot(fpr, tpr, color=colors[name], lw=2.5,
            label=f"{name} (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("    → Saved roc_curves.png")

# ─ Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                linewidths=1, linecolor="white")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("    → Saved confusion_matrices.png")

# ─ Metrics Comparison Bar Chart
metrics = ["accuracy", "auc", "f1", "precision", "recall"]
metric_labels = ["Accuracy", "AUC-ROC", "F1 Score", "Precision", "Recall"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=colors[name], alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Metrics — Model Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    → Saved metrics_comparison.png")

# ─ Feature Importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_names  = X.columns.tolist()
sorted_idx  = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
palette = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_names)))
bars = ax.barh([feat_names[i] for i in sorted_idx[::-1]],
               [importances[i] for i in sorted_idx[::-1]],
               color=palette, edgecolor="white")
for bar in bars:
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    → Saved feature_importance.png")

# ── 6. Hyperparameter Tuning (Random Forest) ─────────────────────────────────
print("\n[6] Hyperparameter Tuning — Random Forest")
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_best  = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]

print(f"    Best params : {grid_search.best_params_}")
print(f"    Best CV AUC : {grid_search.best_score_:.3f}")
print(f"    Test AUC    : {roc_auc_score(y_test, y_proba_best):.3f}")
print(f"    Test Acc    : {accuracy_score(y_test, y_pred_best):.3f}")

# ── 7. Summary Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    name: {
        "Accuracy":  f"{r['accuracy']:.3f}",
        "AUC-ROC":   f"{r['auc']:.3f}",
        "F1 Score":  f"{r['f1']:.3f}",
        "Precision": f"{r['precision']:.3f}",
        "Recall":    f"{r['recall']:.3f}",
        "CV AUC":    f"{r['cv_auc']:.3f}±{r['cv_std']:.3f}",
    }
    for name, r in results.items()
}).T
print(summary.to_string())

best_model_name = max(results, key=lambda k: results[k]["auc"])
print(f"\n  ✅ Best Model: {best_model_name}")
print(f"     AUC-ROC  : {results[best_model_name]['auc']:.3f}")
print(f"     Accuracy : {results[best_model_name]['accuracy']:.3f}")
print("\n  All charts saved. Review PNG files for visual analysis.")
print("=" * 65)
