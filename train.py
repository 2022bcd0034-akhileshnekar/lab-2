import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 6. Save Model
with open("output/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 7. Save Metrics
results = {
    "MSE": mse,
    "R2_Score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

# 8. Print Metrics 
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
