import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 🔥 Set MLflow tracking path (ROOT LEVEL)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Flight Price Prediction")

# 📥 Load dataset
df = pd.read_csv("data/raw/flights.csv")
df = df.dropna()

# 🔄 Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 🎯 Features & target
X = df.drop("price", axis=1)
y = df["price"]

# 💾 Save column order
joblib.dump(X.columns.tolist(), "models/columns.pkl")

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🚀 MLflow Run
with mlflow.start_run(run_name="flight_price_run"):

    print("🚀 MLflow Run Started")

    # 🤖 Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 📊 Predictions
    y_pred = model.predict(X_test)

    # 📉 RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 📝 Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # 📊 Log metric
    mlflow.log_metric("rmse", rmse)

    # 📦 Log model
    mlflow.sklearn.log_model(model, "model")

    print("✅ MLflow logging done")

# 💾 Save model + encoders
joblib.dump(model, "models/flight_price_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print(f"✅ Model trained with RMSE: {rmse}")