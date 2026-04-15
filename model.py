import pandas as pd
import numpy as np

# ── Step 1: Load data ──────────────────────────────────────────────
df = pd.read_csv("car.csv")
print("✅ Data loaded:", df.shape)

# ── Step 2: Check for missing values ──────────────────────────────
print("\nMissing values:")
print(df.isnull().sum())

# ── Step 3: Create new feature — Age of car ───────────────────────
df["Car_Age"] = 2024 - df["Year"]
print("\n✅ Car_Age column added")

# ── Step 4: Encode categorical columns into numbers ───────────────
# ML models only understand numbers, not text like "Petrol" or "Manual"

df["Fuel_Type"] = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type"] = df["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

print("\n✅ Categorical columns encoded")

# ── Step 5: Select features (inputs) and target (output) ──────────
# X = what we feed into the model
# y = what we want the model to predict

X = df[["Present_Price", "Car_Age", "Kms_Driven",
        "Fuel_Type", "Seller_Type", "Transmission", "Owner"]]
y = df["Selling_Price"]

print("\n✅ Features (X):")
print(X.head())

print("\n✅ Target (y):")
print(y.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ── Step 1: Split data into training and testing sets ──────────────
# 80% of data → train the model
# 20% of data → test how well it learned

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data split done")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# ── Step 2: Create the Random Forest model ────────────────────────
# n_estimators = number of decision trees in the forest
# random_state = ensures same results every time you run

model = RandomForestRegressor(n_estimators=100, random_state=42)

# ── Step 3: Train the model ───────────────────────────────────────
print("\n⏳ Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ── Step 4: Save the model to a file ─────────────────────────────
# So we can reuse it in our web app without retraining every time
import joblib
joblib.dump(model, "car_price_model.pkl")
print("✅ Model saved as car_price_model.pkl")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Step 1: Make predictions on test data ─────────────────────────
y_pred = model.predict(X_test)

# ── Step 2: Calculate accuracy metrics ───────────────────────────

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"   R² Score        : {r2:.4f}   (closer to 1.0 = better)")
print(f"   MAE             : {mae:.4f}  (avg error in ₹ Lakhs)")
print(f"   RMSE            : {rmse:.4f} (penalizes big errors more)")

# ── Step 3: Feature importance ────────────────────────────────────
# Which features matter most to the model?

features = ["Present_Price", "Car_Age", "Kms_Driven",
            "Fuel_Type", "Seller_Type", "Transmission", "Owner"]

importances = model.feature_importances_

print("\n🔍 Feature Importance (higher = more influential):")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"   {feat:<16} {bar} {imp:.4f}")

# ── Step 4: Sample predictions vs actual ─────────────────────────
print("\n🔎 Sample Predictions vs Actual (first 8 test rows):")
print(f"   {'Actual':>10} {'Predicted':>10} {'Difference':>12}")
print("   " + "-" * 35)
for actual, predicted in zip(list(y_test[:8]), y_pred[:8]):
    diff = predicted - actual
    print(f"   {actual:>10.2f} {predicted:>10.2f} {diff:>+12.2f}")