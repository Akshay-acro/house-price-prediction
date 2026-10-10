import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


data = pd.read_csv("house_data.csv")

print("Dataset loaded successfully!")
print(data.head())

required_columns = [
    "area",
    "bedrooms",
    "bathrooms",
    "age",
    "location",
    "price"
]

missing_columns = set(required_columns) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

data = data.dropna(subset=required_columns)

if len(data) < 5:
    raise ValueError(
        "The dataset must contain at least 5 complete rows."
    )

X = data[
    ["area", "bedrooms", "bathrooms", "age", "location"]
]

y = data["price"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["location"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining the house price prediction model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Mean Absolute Error: Rs. {mae:,.2f}")
print(f"R2 Score: {r2:.2f}")

joblib.dump(model, "house_model.pkl")

print("\nTrained model saved as house_model.pkl")
print("You can now use this model in app.py!")