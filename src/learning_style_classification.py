# -*- coding: utf-8 -*-
"""
Learning Style Classification using Random Forest

This script performs the machine learning modeling process for classifying
elementary school students' learning styles based on learning habit questionnaire data.

Workflow:
1. Load dataset
2. Validate required columns
3. Clean and preprocess data
4. Encode target label
5. Split dataset
6. Train Random Forest model with hyperparameter tuning
7. Evaluate model performance
8. Save evaluation results and trained model artifact
"""

from pathlib import Path
import os
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# Project Path Setup
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "Data_siswa.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")


def print_section(title: str) -> None:
    """Print a formatted section title."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_text_report(content: str, filename: str) -> None:
    """Save text content into the results directory."""
    output_path = RESULTS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Saved: {output_path}")


# ============================================================
# Data Loading
# ============================================================

print_section("Data Loading")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Please make sure Data_siswa.csv is located in the data/ folder."
    )

df = pd.read_csv(DATASET_PATH, sep=None, engine="python")

# Remove BOM character if it exists in column names
df.columns = df.columns.str.replace("\ufeff", "", regex=False)

print("Dataset path:", DATASET_PATH)
print("Dataset shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nPreview dataset:")
print(df.head())


# ============================================================
# Dataset Validation
# ============================================================

print_section("Dataset Validation")

required_columns = [
    "Intensitas_Belajar",
    "Fokus_Belajar",
    "Disiplin_Belajar",
    "Interaksi_Belajar",
    "Teknologi",
    "Lingkungan_Belajar",
    "Waktu_Efektif",
    "Gaya_Belajar",
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"Required columns not found in dataset: {missing_cols}")

print("All required columns are available.")


# ============================================================
# Variable Definition
# ============================================================

print_section("Variable Definition")

feature_cols = [
    "Intensitas_Belajar",
    "Fokus_Belajar",
    "Disiplin_Belajar",
    "Interaksi_Belajar",
    "Teknologi",
    "Lingkungan_Belajar",
    "Waktu_Efektif",
]

target_col = "Gaya_Belajar"

variable_table = pd.DataFrame({
    "No": range(1, len(feature_cols) + 2),
    "Variable": feature_cols + [target_col],
    "Role": ["Feature"] * len(feature_cols) + ["Target"],
    "Data Type": ["Categorical"] * (len(feature_cols) + 1),
})

print(variable_table.to_string(index=False))
variable_table.to_csv(RESULTS_DIR / "variable_table.csv", index=False)


# ============================================================
# Data Cleaning
# ============================================================

print_section("Data Cleaning")

df_model = df.copy()

print("Initial data shape:", df_model.shape)

# Replace blank strings with NaN
df_model = df_model.replace(r"^\s*$", np.nan, regex=True)

print("\nMissing values before cleaning:")
print(df_model.isna().sum())

duplicate_count = df_model.duplicated().sum()
df_model = df_model.drop_duplicates().reset_index(drop=True)

print(f"\nNumber of duplicate rows removed: {duplicate_count}")
print("Data shape after removing duplicates:", df_model.shape)

# Remove identity attributes if available
identity_columns = [
    "ID_Siswa",
    "Sekolah",
    "Nomor_Absen",
    "Nama",
    "Kelamin",
    "Kelas",
]

df_model = df_model.drop(columns=identity_columns, errors="ignore")

print("\nColumns after removing identity attributes:")
print(df_model.columns.tolist())

# Select only model features and target
df_model = df_model[feature_cols + [target_col]]

# Impute missing values in feature columns
imputer = SimpleImputer(strategy="most_frequent")
df_model[feature_cols] = imputer.fit_transform(df_model[feature_cols])

print("\nMissing values after imputation:")
print(df_model.isna().sum())

print("\nData shape after cleaning:", df_model.shape)
print("\nPreview cleaned data:")
print(df_model.head())

df_model.to_csv(RESULTS_DIR / "cleaned_data_preview.csv", index=False)


# ============================================================
# Target Encoding
# ============================================================

print_section("Target Encoding")

X = df_model[feature_cols].copy()
y_raw = df_model[target_col].astype(str).copy()

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)

mapping_df = pd.DataFrame({
    "Learning Style Class": target_encoder.classes_,
    "Numeric Code": range(len(target_encoder.classes_)),
})

print("Target label mapping:")
print(mapping_df.to_string(index=False))

mapping_df.to_csv(RESULTS_DIR / "target_label_mapping.csv", index=False)


# ============================================================
# Train-Test Split
# ============================================================

print_section("Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

split_table = pd.DataFrame({
    "Data": ["Train", "Test"],
    "Total": [len(X_train), len(X_test)],
    "Percentage": [
        f"{len(X_train) / len(X) * 100:.0f}%",
        f"{len(X_test) / len(X) * 100:.0f}%",
    ],
})

print("Train-test split summary:")
print(split_table.to_string(index=False))

split_table.to_csv(RESULTS_DIR / "train_test_split_summary.csv", index=False)


# ============================================================
# Class Distribution and Cross-Validation Setup
# ============================================================

print_section("Class Distribution and Cross-Validation Setup")

train_dist = pd.Series(target_encoder.inverse_transform(y_train)).value_counts()
test_dist = pd.Series(target_encoder.inverse_transform(y_test)).value_counts()

distribution_table = pd.DataFrame({
    "Class": sorted(target_encoder.classes_),
    "Train": [int(train_dist.get(k, 0)) for k in sorted(target_encoder.classes_)],
    "Test": [int(test_dist.get(k, 0)) for k in sorted(target_encoder.classes_)],
})

print("Class distribution in train and test data:")
print(distribution_table.to_string(index=False))

distribution_table.to_csv(RESULTS_DIR / "class_distribution_train_test.csv", index=False)

class_counts = pd.Series(y_train).value_counts().sort_index()
min_class = int(class_counts.min())
max_class = int(class_counts.max())
imbalance_ratio = (max_class / min_class) if min_class > 0 else 0

print("\nClass distribution in training data:")
print(class_counts)
print(f"\nMinimum class count: {min_class}")
print(f"Maximum class count: {max_class}")
print(f"Imbalance ratio    : {imbalance_ratio:.2f}")

cv_folds = max(2, min(5, min_class))

cv = StratifiedKFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=42,
)

print(f"\nNumber of cross-validation folds: {cv_folds}")


# ============================================================
# Preprocessing Preview
# ============================================================

print_section("Preprocessing Preview")

preprocess_preview = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

X_transformed = preprocess_preview.fit_transform(X)
transformed_df = pd.DataFrame(X_transformed, columns=feature_cols)

print("Preview of transformed feature data:")
print(transformed_df.head())

transformed_df.head().to_csv(RESULTS_DIR / "transformed_feature_preview.csv", index=False)


# ============================================================
# Random Forest Modeling Pipeline
# ============================================================

print_section("Random Forest Modeling")

rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ("model", rf_model),
])

print("Model pipeline steps:")
for step_name, _ in pipeline.steps:
    print("-", step_name)


# ============================================================
# Hyperparameter Tuning
# ============================================================

print_section("Hyperparameter Tuning")

param_dist = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [None, 10, 15, 20],
    "model__min_samples_split": [2, 3, 5],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}

param_desc = {
    "model__n_estimators": "Number of decision trees in the Random Forest",
    "model__max_depth": "Maximum depth of each tree",
    "model__min_samples_split": "Minimum number of samples required to split an internal node",
    "model__min_samples_leaf": "Minimum number of samples required to be at a leaf node",
    "model__max_features": "Number of features considered when looking for the best split",
}

param_table = pd.DataFrame([
    {
        "Parameter": param.replace("model__", ""),
        "Tested Values": ", ".join([str(v) for v in values]),
        "Description": param_desc.get(param, "-"),
    }
    for param, values in param_dist.items()
])

print("Hyperparameter search space:")
print(param_table.to_string(index=False))

param_table.to_csv(RESULTS_DIR / "hyperparameter_search_space.csv", index=False)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

print("\nHyperparameter tuning started...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
for param_name, param_value in search.best_params_.items():
    print(f"{param_name}: {param_value}")

print(f"\nBest CV F1-macro: {search.best_score_:.4f}")

best_params_df = pd.DataFrame([
    {"Parameter": key.replace("model__", ""), "Best Value": value}
    for key, value in search.best_params_.items()
])

best_params_df.to_csv(RESULTS_DIR / "best_hyperparameters.csv", index=False)


# ============================================================
# Model Evaluation
# ============================================================

print_section("Model Evaluation")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f"Number of test samples: {len(y_test)}")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"F1-score : {f1_macro:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_,
    digits=4,
)

print("\nClassification Report:")
print(report)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score"],
    "Score": [accuracy, f1_macro],
})

metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

save_text_report(report, "classification_report.txt")


# ============================================================
# Confusion Matrix
# ============================================================

print_section("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()

confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()

print("Confusion matrix saved to:", confusion_matrix_path)
print("Classes:", list(target_encoder.classes_))


# ============================================================
# Feature Importance
# ============================================================

print_section("Feature Importance")

feature_importances = best_model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": feature_importances,
}).sort_values(by="Importance", ascending=False)

print("Feature importance:")
print(feature_importance_df.to_string(index=False))

feature_importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"],
)

plt.title("Feature Importance - Random Forest", fontsize=12)
plt.xlabel("Feature", fontsize=11)
plt.ylabel("Importance Score", fontsize=11)
plt.xticks(rotation=30, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

feature_importance_path = RESULTS_DIR / "feature_importance.png"
plt.savefig(feature_importance_path, dpi=300)
plt.close()

print("Feature importance chart saved to:", feature_importance_path)


# ============================================================
# Cross-Validation Scores Plot
# ============================================================

print_section("Cross-Validation Scores")

mean_scores = search.cv_results_["mean_test_score"]

cv_scores_df = pd.DataFrame({
    "Iteration": range(1, len(mean_scores) + 1),
    "Mean CV F1-Score": mean_scores,
})

cv_scores_df.to_csv(RESULTS_DIR / "cv_scores.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(
    range(1, len(mean_scores) + 1),
    mean_scores,
    marker="o",
    linestyle="-",
)

plt.xlabel("RandomizedSearch Iteration", fontsize=11)
plt.ylabel("Mean CV F1-Score", fontsize=11)
plt.title("Cross-Validation Performance during RandomizedSearchCV", fontsize=12)

best_score = search.best_score_
plt.axhline(
    y=best_score,
    linestyle="--",
    linewidth=1,
    label=f"Best CV F1 = {best_score:.3f}",
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

cv_scores_path = RESULTS_DIR / "cv_scores.png"
plt.savefig(cv_scores_path, dpi=300)
plt.close()

print("Cross-validation score chart saved to:", cv_scores_path)


# ============================================================
# Save Model Artifact
# ============================================================

print_section("Save Model Artifact")

artifact = {
    "model": best_model,
    "feature_cols": feature_cols,
    "target_encoder": target_encoder,
    "target_classes": target_encoder.classes_,
}

model_path = MODELS_DIR / "model_rf_gaya_belajar.joblib"
joblib.dump(artifact, model_path)

print(f"Model artifact saved to: {model_path}")


# ============================================================
# Final Summary
# ============================================================

print_section("Final Summary")

print("Modeling process completed successfully.")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"F1-score : {f1_macro:.4f}")
print(f"Results saved in: {RESULTS_DIR}")
print(f"Model saved in  : {model_path}")