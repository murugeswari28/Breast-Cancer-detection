import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data.csv")

print("Dataset loaded successfully!")
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

FEATURES = [
    "radius_mean",
    "perimeter_mean",
    "area_mean"
]

TARGET = "diagnosis"


# Check required columns
for column in FEATURES + [TARGET]:
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset."
        )


# --------------------------------------------------
# 3. PREPARE INPUT AND OUTPUT
# --------------------------------------------------

X = df[FEATURES]

# M = Malignant
# B = Benign
y = df[TARGET].map({
    "M": 0,
    "B": 1
})


# Remove invalid rows if any
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]


# --------------------------------------------------
# 4. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. CREATE RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. TEST MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------------------")
print("MODEL TRAINING COMPLETED")
print("--------------------------------")

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nmodel.pkl created successfully!")
print("Training completed.")