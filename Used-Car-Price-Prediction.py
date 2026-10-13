
# %% =====================================
# IMPORTS
# ========================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# %% =====================================
# LOAD DATA
# ========================================

df = pd.read_csv("car_sales_data.csv")

print(df.head())
print(df.shape)


# %% =====================================
# DATA CHECK & CLEANING
# ========================================

df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

print("\nShape after removing duplicates:")
print(df.shape)


# %% =====================================
# OUTLIER CHECK
# ========================================

numeric_cols = [
    "Engine size",
    "Year of manufacture",
    "Mileage",
    "Price"
]

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# Count potential outliers using the IQR method
for col in ["Engine size", "Mileage", "Price"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"{col}: {len(outliers)} potential outliers")


# %% =====================================
# EXPLORATORY SCATTERPLOTS
# ========================================

sns.scatterplot(data=df, x="Mileage", y="Price")
plt.title("Mileage vs Price")
plt.show()

sns.scatterplot(data=df, x="Engine size", y="Price")
plt.title("Engine size vs Price")
plt.show()


# %% =====================================
# SPLIT THE DATA
# ========================================

feature_cols = [
    "Engine size",
    "Year of manufacture",
    "Mileage",
    "Manufacturer",
    "Model",
    "Fuel type"
]

X = df[feature_cols].copy()
y = df["Price"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)


# %% =====================================
# PREPROCESSING
# ========================================

# Numerical features that need a log transformation
log_features = ["Engine size", "Mileage"]

# Numerical feature that only needs scaling
year_features = ["Year of manufacture"]

# Categorical features that need encoding
categorical_features = [
    "Manufacturer",
    "Model",
    "Fuel type"
]


# Log transformation followed by scaling
log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(
        np.log1p,
        feature_names_out="one-to-one"
    )),
    ("scaler", StandardScaler())
])


# Scale the year of manufacture
year_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Convert categorical values into numerical columns
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# Combine all preprocessing steps
preprocessor = ColumnTransformer([
    ("log_features", log_pipeline, log_features),
    ("year_features", year_pipeline, year_features),
    ("categorical_features", categorical_pipeline,
     categorical_features)
])


# Learn preprocessing rules from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the learned rules to test data
X_test_processed = preprocessor.transform(X_test)


# Convert processed features into NumPy arrays
X_train_np = np.asarray(
    X_train_processed, dtype=np.float32
)

X_test_np = np.asarray(
    X_test_processed, dtype=np.float32
)

# Prepare target values
y_train_np = y_train.to_numpy(
    dtype=np.float32
).reshape(-1, 1)

y_test_np = y_test.to_numpy(
    dtype=np.float32
).reshape(-1, 1)


print("\nProcessed training features:", X_train_np.shape)
print("Processed testing features:", X_test_np.shape)


# %% =====================================
# PYTORCH LINEAR REGRESSION
# ========================================

# Convert NumPy arrays into PyTorch tensors
X_train_torch = torch.from_numpy(X_train_np)
X_test_torch = torch.from_numpy(X_test_np)

# Standardize the target using training prices only
y_mean = y_train_np.mean()
y_std = y_train_np.std()

if y_std == 0:
    raise ValueError("All training prices are identical.")

y_train_scaled = (y_train_np - y_mean) / y_std

y_train_torch = torch.from_numpy(
    y_train_scaled.astype(np.float32)
)


# Initialize model parameters
torch.manual_seed(42)

num_features = X_train_torch.shape[1]

w = torch.zeros(
    num_features, 1, requires_grad=True
)

b = torch.zeros(1, requires_grad=True)


# Training settings
num_epochs = 1000
learning_rate = 0.01


# %% =====================================
# TRAINING LOOP
# ========================================

for epoch in range(num_epochs):

    # Predict training prices
    y_pred = X_train_torch @ w + b

    # Calculate mean squared error
    loss = torch.mean(
        (y_pred - y_train_torch) ** 2
    )

    # Calculate gradients
    loss.backward()

    # Update weights and bias
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.4f}"
        )


# %% =====================================
# PYTORCH PREDICTIONS
# ========================================

with torch.no_grad():

    # Predict on unseen test data
    y_test_pred_scaled = X_test_torch @ w + b

    # Convert predictions back to original price scale
    y_test_pred_torch = (
        y_test_pred_scaled.numpy() * y_std + y_mean
    ).ravel()


print("\nFirst five PyTorch predictions:")
print(y_test_pred_torch[:5])

print("\nFirst five actual prices:")
print(y_test_np[:5].ravel())


# %% =====================================
# PYTORCH EVALUATION
# ========================================

r2_torch = r2_score(
    y_test_np.ravel(),
    y_test_pred_torch
)

mae_torch = mean_absolute_error(
    y_test_np.ravel(),
    y_test_pred_torch
)

rmse_torch = np.sqrt(
    mean_squared_error(
        y_test_np.ravel(),
        y_test_pred_torch
    )
)

print("\nPyTorch Test Results")
print("R²:", round(r2_torch, 4))
print("MAE:", round(mae_torch, 2))
print("RMSE:", round(rmse_torch, 2))


# %% =====================================
# SCIKIT-LEARN LINEAR REGRESSION
# ========================================

reg = LinearRegression()

# Train on the same processed training data
reg.fit(X_train_np, y_train_np.ravel())

# Predict on the test data
y_test_pred_sklearn = reg.predict(X_test_np)


print("\nScikit-learn coefficients:")
print(reg.coef_)

print("\nScikit-learn intercept:")
print(reg.intercept_)


# %% =====================================
# SCIKIT-LEARN EVALUATION
# ========================================

r2_sklearn = r2_score(
    y_test_np.ravel(),
    y_test_pred_sklearn
)

mae_sklearn = mean_absolute_error(
    y_test_np.ravel(),
    y_test_pred_sklearn
)

rmse_sklearn = np.sqrt(
    mean_squared_error(
        y_test_np.ravel(),
        y_test_pred_sklearn
    )
)

print("\nScikit-learn Test Results")
print("R²:", round(r2_sklearn, 4))
print("MAE:", round(mae_sklearn, 2))
print("RMSE:", round(rmse_sklearn, 2))


# %% =====================================
# COMPARE BOTH MODELS
# ========================================

results = pd.DataFrame({
    "Model": ["PyTorch", "Scikit-learn"],
    "R²": [r2_torch, r2_sklearn],
    "MAE": [mae_torch, mae_sklearn],
    "RMSE": [rmse_torch, rmse_sklearn]
})

print("\nModel comparison:")
print(results)


# %% =====================================
# ACTUAL VS PREDICTED PRICES
# ========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_np.ravel(),
    y_test_pred_torch,
    alpha=0.6,
    label="PyTorch"
)

plt.scatter(
    y_test_np.ravel(),
    y_test_pred_sklearn,
    alpha=0.6,
    label="Scikit-learn"
)

min_price = min(
    y_test_np.min(),
    y_test_pred_torch.min(),
    y_test_pred_sklearn.min()
)

max_price = max(
    y_test_np.max(),
    y_test_pred_torch.max(),
    y_test_pred_sklearn.max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    "r--",
    label="Perfect predictions"
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.show()


# %% =====================================
# RESIDUAL PLOT: PYTORCH
# ========================================

residuals = y_test_np.ravel() - y_test_pred_torch

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_pred_torch,
    residuals,
    alpha=0.6
)

plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("PyTorch Residual Plot")
plt.show()


