import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib

print("🔄 Loading data...")
df = pd.read_csv("data/telco_churn.csv")

# --- Clean Data ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop(columns=["customerID"], inplace=True)

# --- Encode Categorical Columns ---
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# --- Features & Target ---
X = df.drop(columns=["Churn"])
y = df["Churn"]
feature_names = list(X.columns)

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data prepared\n")

# --- Train & Compare Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_auc = 0
best_name = ""

print("=== Model Comparison ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"📌 {name}")
    print(f"   AUC Score : {auc:.4f}")
    print(f"   Report:\n{classification_report(y_test, y_pred)}")
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} | AUC = {best_auc:.4f}")

# --- Save Model, Scaler, Feature Names ---
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
print("✅ Model saved as model.pkl")

# --- Plot ROC Curve ---
fpr, tpr, _ = roc_curve(y_test,
                          best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"AUC = {best_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("✅ ROC curve saved")

# --- Plot Confusion Matrix ---
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Retained", "Churned"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("✅ Confusion matrix saved")

# --- Feature Importance (if applicable) ---
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_,
                             index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    importances.head(10).plot(kind="bar", color="steelblue")
    plt.title("Top 10 Feature Importances")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("✅ Feature importance chart saved")

print("\n🎉 Training complete!")