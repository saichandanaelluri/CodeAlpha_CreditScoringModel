
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
data = pd.read_csv("dataset.csv")
data.columns = data.columns.str.strip()

print("Column names:", data.columns.tolist())
print("\nDataset:")
print(data.head())

print("\nDataset Information:")
data.info()

# 2. Define the target column (correct lowercase name)
target = "creditworthy"

if target not in data.columns:
    raise ValueError(
        f"Target '{target}' not found. "
        f"Available columns: {data.columns.tolist()}"
    )

# Remove rows with missing target values
data = data.dropna(subset=[target]).copy()

# Ensure target is binary (0 and 1)
if data[target].nunique() != 2 or not set(
    data[target].unique()
).issubset({0, 1}):
    raise ValueError(
        "The creditworthy column must contain only 0 and 1, "
        "with both classes present."
    )

# 3. Separate features and target
X = data.drop(columns=[target])
y = data[target].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

# 4. Identify numerical and categorical features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both training and testing sets must contain both classes. "
        "Check the number of rows belonging to each class."
    )

# 7. Create models
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

# 8. Train and evaluate models
results = []
predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    results.append(metrics)

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric:10}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

# 9. Model performance comparison
results_df = pd.DataFrame(results).set_index("Model")

print("\nModel Performance Comparison:")
print(results_df.round(4))

results_df.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1.05)
)

plt.title("Logistic Regression vs Random Forest")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 10. Random Forest confusion matrix
rf_pred = predictions["Random Forest"]

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=[0, 1]
)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Creditworthy (0)", "Creditworthy (1)"],
    yticklabels=["Not Creditworthy (0)", "Creditworthy (1)"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 11. Random Forest feature importance
rf_classifier = models["Random Forest"].named_steps["classifier"]
fitted_preprocessor = models["Random Forest"].named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.Series(
    rf_classifier.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

top_features = importance.head(15).sort_values()

plt.figure(figsize=(9, 5))
sns.barplot(
    x=top_features.values,
    y=top_features.index
)

plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
print("\nCredit Scoring Model completed successfully!")
