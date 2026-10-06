import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv("dataset.csv")
X = data.drop("creditworthy", axis=1)
y = data["creditworthy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
def evaluate_model(name, y_test, predictions, probabilities):
    print("\n", name)
    print("-" * 30)
    print("Accuracy :", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall :", recall_score(y_test, predictions, zero_division=0))
    print("F1-Score :", f1_score(y_test, predictions, zero_division=0))
    print("ROC-AUC :", roc_auc_score(y_test, probabilities))
evaluate_model("Logistic Regression", y_test, logistic_pred, logistic_prob)
evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
print("\nCredit Scoring Model completed successfully!")