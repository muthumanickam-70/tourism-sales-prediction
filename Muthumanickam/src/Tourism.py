# ================================================================
# Data Mining for Tourism Demand Prediction
# ================================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# Load Dataset
# ================================================================

print("\nLoading Tourism Dataset...\n")

data = pd.read_csv("daily_tourism_demand_forecasting_2000_2015.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# ================================================================
# Generate Target
# ================================================================

np.random.seed(42)

data["Visitor_Count"] = (
    data["Hotel_Occupancy"] * 120 +
    data["Flight_Arrivals"] * 8 +
    data["Average_Temperature"] * 20 +
    data["Economic_Index"] * 500
)

# ================================================================
# Filter Destination
# ================================================================

data = data[data["Destination"] == "Paris"]

# ================================================================
# Date Features
# ================================================================

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date")

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day_of_Week"] = data["Date"].dt.dayofweek

# ================================================================
# Lag Features
# ================================================================

data["Lag1"] = data["Visitor_Count"].shift(1)
data["Lag7"] = data["Visitor_Count"].shift(7)
data["Lag14"] = data["Visitor_Count"].shift(14)

data = data.dropna()

# ================================================================
# Encode Event
# ================================================================

encoder = LabelEncoder()

data["Major_Event"] = encoder.fit_transform(data["Major_Event"])

# ================================================================
# Remove Columns
# ================================================================

data = data.drop(columns=["Date","Destination"])

# ================================================================
# Feature Selection
# ================================================================

X = data.drop(columns=["Visitor_Count"])
y = data["Visitor_Count"]

corr = X.corrwith(y).abs()

selected_features = corr.sort_values(ascending=False).head(10).index.tolist()

print("Selected Features:", selected_features)

X = X[selected_features]

# ================================================================
# Normalization
# ================================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled,columns=X.columns)

# ================================================================
# Train Test Split
# ================================================================

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ================================================================
# XGBoost Model
# ================================================================

print("\nOptimizing Model...\n")

param_grid = {

    "n_estimators":[300,500,700],
    "max_depth":[4,6,8],
    "learning_rate":[0.01,0.03,0.05],
    "subsample":[0.8,0.9,1],
    "colsample_bytree":[0.8,0.9,1]

}

xgb = XGBRegressor(random_state=42)

search = RandomizedSearchCV(

    xgb,
    param_grid,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1

)

search.fit(X_train,y_train)

model = search.best_estimator_

print("Best Parameters:",search.best_params_)

# ================================================================
# Prediction
# ================================================================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print("RMSE:",rmse)
print("R2:",r2)

# ================================================================
# Feature Importance
# ================================================================

plt.figure(figsize=(8,6))

sns.barplot(x=model.feature_importances_,y=selected_features)

plt.title("Feature Importance")

plt.show()

# ================================================================
# Save Model
# ================================================================

joblib.dump(model,"tourism_xgb_model.pkl")
joblib.dump(encoder,"event_encoder.pkl")
joblib.dump(selected_features,"selected_features.pkl")
joblib.dump(scaler,"tourism_scaler.pkl")

print("\nModel Saved Successfully!")