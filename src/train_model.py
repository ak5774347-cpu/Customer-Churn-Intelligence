
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("data/customer_data.csv")

# 2. Select features and target
X = df.drop(columns=["CustomerID", "Churn", "TotalRevenue"])
y = df["Churn"]

# 3. Identify categorical and numerical columns
categorical = ["Contract", "PaymentMethod"]

numerical = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "SupportCalls",
    "UsageHours"
]

# 4. Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ]
)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Build the model with balanced class weights
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
# Get churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Use a lower threshold to identify more potential churners
threshold = 0.40
y_pred = (y_prob >= threshold).astype(int)

# 9. Evaluate the model
print("\n========== MODEL EVALUATION ==========")

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed (0)", "Churned (1)"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

