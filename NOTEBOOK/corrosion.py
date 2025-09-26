# -------------------------------
# 1. Imports and Dataset
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("DATA/corrosion_synthetic_timeseries.csv")
print(df.head())

# Add pH² to capture U-shaped effect
df["pH_squared"] = df["pH"]**2

# Define features (5 original + derived pH²)
features = ["pH","pH_squared","temperature_C","chloride_ppm","dissolved_oxygen_mgL","flow_velocity_mps"]
X = df[features]
y = df["corrosion_rate_mm_per_yr"]

# -------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------
# Corrosion vs pH
plt.scatter(df["pH"], df["corrosion_rate_mm_per_yr"])
plt.xlabel("pH")
plt.ylabel("Corrosion Rate (mm/yr)")
plt.show()

# Corrosion vs Temperature
plt.scatter(df["temperature_C"], df["corrosion_rate_mm_per_yr"])
plt.xlabel("Temperature (°C)")
plt.ylabel("Corrosion Rate (mm/yr)")
plt.show()

# Correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# -------------------------------
# 3. Regression Model with pH²
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² score:", model.score(X_test, y_test))

# -------------------------------
# 4. Diagnostics
# -------------------------------
# Predicted vs Actual
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Corrosion Rate (mm/yr)")
plt.ylabel("Predicted Corrosion Rate (mm/yr)")
plt.title("Regression: Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Corrosion Rate")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.show()

# -------------------------------
# 5. Partial Effects
# -------------------------------
baseline = X_train.median().to_dict()

# Partial Effect of pH
ph_grid = np.linspace(df["pH"].min(), df["pH"].max(), 80)
rows = []
for ph in ph_grid:
    r = baseline.copy()
    r["pH"] = ph
    r["pH_squared"] = ph**2
    rows.append(r)

grid_df = pd.DataFrame(rows)[features]
yhat = model.predict(grid_df)

plt.figure(figsize=(7,5))
plt.plot(ph_grid, yhat, color="blue")
plt.xlabel("pH (others fixed at median)")
plt.ylabel("Predicted Corrosion Rate (mm/yr)")
plt.title("Partial Effect of pH on Corrosion Rate")
plt.grid(True)
plt.show()

# Partial effects of other variables
variables = ["temperature_C","chloride_ppm","dissolved_oxygen_mgL","flow_velocity_mps"]
plt.figure(figsize=(12,8))

for i, var in enumerate(variables, 1):
    grid = np.linspace(df[var].min(), df[var].max(), 80)
    rows = []
    for val in grid:
        r = baseline.copy()
        r[var] = val
        rows.append(r)
    grid_df = pd.DataFrame(rows)[features]
    yhat = model.predict(grid_df)

    plt.subplot(2, 2, i)
    plt.plot(grid, yhat, color="blue")
    plt.xlabel(var)
    plt.ylabel("Predicted corrosion rate (mm/yr)")
    plt.title(f"Effect of {var} on corrosion")
    plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# 6. Scenario Analysis with RUL
# -------------------------------
scenarios = [
    {"Scenario": "Seawater pipeline", "pH": 8.0, "temperature_C": 30,
     "chloride_ppm": 1000, "dissolved_oxygen_mgL": 6, "flow_velocity_mps": 1.0},
    {"Scenario": "Acidic soil", "pH": 4.5, "temperature_C": 25,
     "chloride_ppm": 200, "dissolved_oxygen_mgL": 3, "flow_velocity_mps": 0.5},
    {"Scenario": "Neutral freshwater", "pH": 7.0, "temperature_C": 20,
     "chloride_ppm": 50, "dissolved_oxygen_mgL": 8, "flow_velocity_mps": 0.3}
]

allowable_loss = 1.0  # mm
results = []
for s in scenarios:
    s["pH_squared"] = s["pH"]**2
    X_scenario = pd.DataFrame([s])[features]  # enforce order
    pred_rate = float(model.predict(X_scenario)[0])
    pred_rate = max(0.0, pred_rate)  # clip negatives
    RUL = (allowable_loss / pred_rate) if pred_rate > 0 else float("inf")
    results.append({
        "Scenario": s["Scenario"],
        "Predicted corrosion (mm/yr)": round(pred_rate, 3),
        "Remaining Useful Life (years)": (round(RUL, 1) if pred_rate > 0 else "∞")
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# -------------------------------
# 7. Heatmap: Temperature × Chloride
# -------------------------------
baseline = X_train.median().to_dict()

temp_grid = np.linspace(df["temperature_C"].min(), df["temperature_C"].max(), 50)
cl_grid   = np.linspace(df["chloride_ppm"].min(), df["chloride_ppm"].max(), 50)

rows = []
for t in temp_grid:
    for cl in cl_grid:
        r = baseline.copy()
        r["temperature_C"] = t
        r["chloride_ppm"] = cl
        rows.append(r)

grid_df = pd.DataFrame(rows)[features]  # enforce order
yhat = model.predict(grid_df)
Z = yhat.reshape(len(temp_grid), len(cl_grid))

plt.figure(figsize=(8,6))
plt.imshow(Z.T, origin="lower", 
           extent=[temp_grid.min(), temp_grid.max(), cl_grid.min(), cl_grid.max()],
           aspect="auto", cmap="inferno")
plt.colorbar(label="Predicted corrosion rate (mm/yr)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Chloride (ppm)")
plt.title("Predicted Corrosion: Temperature × Chloride")
plt.show()
