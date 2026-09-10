"""
A Comparative Study of 5 Supervised Learning Algorithms
=========================================================
Dataset : Breast Cancer Wisconsin (Diagnostic) Dataset
Source  : UCI Machine Learning Repository (bundled in scikit-learn:
          sklearn.datasets.load_breast_cancer). 569 samples, 30 numeric
          features computed from digitized images of fine needle aspirate
          (FNA) of breast masses. Binary classification target:
          malignant (0) vs benign (1).

This script trains and evaluates 5 distinct supervised classifiers on the
identical train/test split and identical preprocessing pipeline, then
produces a ranked comparison table and chart.

Algorithms compared:
  1. Logistic Regression   (linear, parametric)
  2. Decision Tree         (non-linear, axis-aligned splits)
  3. K-Nearest Neighbors   (non-linear, instance-based, distance-sensitive)
  4. Support Vector Machine (RBF kernel, non-linear, margin-based)
  5. Gaussian Naive Bayes  (probabilistic, strong independence assumption)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
target_col = "target"  # 0 = malignant, 1 = benign

print("Dataset shape:", df.shape)
print("Class balance:\n", df[target_col].value_counts())
print("Missing values total:", df.isnull().sum().sum())

# ---------------------------------------------------------------------
# 2. CLEAN / PREPROCESS
#    - Handle missing values (none present, but pipeline is built to
#      handle them generically so the workflow generalizes to messier
#      real-world data).
#    - All 30 features here are already numeric (no categorical columns
#      in this dataset), so encoding is a no-op step but is included in
#      the pipeline for completeness/documentation purposes.
#    - Split into train/test BEFORE scaling to avoid data leakage.
# ---------------------------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numeric columns: {len(numeric_cols)} | Categorical columns: {len(categorical_cols)}")

# Impute missing values (median for numeric) - defensive, dataset has none.
imputer = SimpleImputer(strategy="median")
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# One-hot encode any categoricals (none exist here, but kept for generality).
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Same train/test split (stratified on target to preserve class balance)
# used identically for every model.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features. Fit the scaler ONLY on the training data, then transform
# both train and test - this prevents test-set information from leaking
# into the preprocessing step.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

# Distance-based / gradient-based models (Logistic Regression, KNN, SVM)
# need scaled features. Tree-based models (Decision Tree) are scale-invariant
# by construction (splits are based on thresholds per feature, unaffected by
# monotonic scaling). Gaussian Naive Bayes assumes per-feature Gaussian
# distributions - scaling doesn't change its decision boundary mathematically,
# so it is trained on the same scaled data for a clean, uniform pipeline.
MODELS_NEEDING_SCALED_INPUT = {
    "Logistic Regression": True,
    "Decision Tree": False,
    "K-Nearest Neighbors": True,
    "Support Vector Machine (RBF)": True,
    "Gaussian Naive Bayes": True,
}

# ---------------------------------------------------------------------
# 3. DEFINE MODELS (identical train/test data, only the algorithm differs)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Support Vector Machine (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Gaussian Naive Bayes": GaussianNB(),
}

# ---------------------------------------------------------------------
# 4. TRAIN + 5. EVALUATE
# ---------------------------------------------------------------------
results = []
confusion_matrices = {}

for name, model in models.items():
    use_scaled = MODELS_NEEDING_SCALED_INPUT[name]
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Input Features": "Scaled" if use_scaled else "Raw (unscaled)",
    })
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
results_df.insert(0, "Rank", results_df.index + 1)

print("\n=== COMPARISON TABLE (ranked by F1-Score) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------
# 6. SAVE COMPARISON TABLE + CHART
# ---------------------------------------------------------------------
results_df.to_csv("results/comparison_table.csv", index=False)

# Save confusion matrices for reference/appendix
with open("results/confusion_matrices.json", "w") as f:
    json.dump({k: v.tolist() for k, v in confusion_matrices.items()}, f, indent=2)

# --- Bar chart: all 4 metrics per model, grouped ---
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
plot_df = results_df.melt(
    id_vars="Model", value_vars=metrics_to_plot,
    var_name="Metric", value_name="Score"
)

plt.figure(figsize=(12, 6.5))
order = results_df.sort_values("F1-Score", ascending=False)["Model"].tolist()
ax = sns.barplot(
    data=plot_df, x="Model", y="Score", hue="Metric",
    order=order, palette="viridis"
)
plt.title("Comparison of 5 Supervised Learning Algorithms\nBreast Cancer Wisconsin (Diagnostic) Dataset",
          fontsize=13, fontweight="bold")
plt.ylim(0.80, 1.02)
plt.ylabel("Score")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/comparison_chart.png", dpi=160)
plt.close()

# --- Simple ranking bar chart on F1-Score alone (the headline chart) ---
plt.figure(figsize=(9, 5.5))
colors = sns.color_palette("viridis", len(results_df))
bars = plt.barh(results_df["Model"][::-1], results_df["F1-Score"][::-1], color=colors[::-1])
plt.xlim(0.85, 1.0)
plt.xlabel("F1-Score (test set)")
plt.title("Model Ranking by F1-Score", fontsize=13, fontweight="bold")
for bar, val in zip(bars, results_df["F1-Score"][::-1]):
    plt.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
              va="center", fontsize=10)
plt.tight_layout()
plt.savefig("results/ranking_chart.png", dpi=160)
plt.close()

print("\nSaved: results/comparison_table.csv, results/comparison_chart.png, "
      "results/ranking_chart.png, results/confusion_matrices.json")
