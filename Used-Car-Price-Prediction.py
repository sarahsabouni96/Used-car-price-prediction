# %% =====================================
# IMPORTS
# ========================================
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#%% =====================================
# LOAD DATA
# ========================================
df = pd.read_csv(r"car_sales_data.csv")

df.head()
print(df)

# %% =====================================
# DATA CHECK & CLEANING
# ========================================
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()

#%% check duplicates
df.duplicated().sum()

# Drop duplicates
df = df.drop_duplicates()
print(df.shape)

# %% =====================================
# OUTLIER CHECK (IQR method + boxplots)
# ========================================

numeric_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

numeric_cols = ['Engine size', 'Mileage', 'Price']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)   # 25th percentile
    Q3 = df[col].quantile(0.75)   # 75th percentile
    IQR = Q3 - Q1                 # Interquartile range

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

# %% =====================================
# EXPLORATORY SCATTERPLOTS
# ========================================


sns.scatterplot(x=df['Mileage'], y=df['Price'])
plt.title("Mileage vs Price")
plt.show()

sns.scatterplot(x=df['Engine size'], y=df['Price'])
plt.title("Engine size vs Price")
plt.show()



# %% =====================================
# FEATURE ENGINEERING
# ========================================

# One-hot encode 'Fuel type'
df = df.copy()

# One-hot encode Fuel type
df = pd.get_dummies(df, columns=['Fuel type'], drop_first=False)

# Convert any boolean dummy columns to integers (0/1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Target encoding for Manufacturer & Model
df['Manufacturer_encoded'] = df.groupby('Manufacturer')['Price'].transform('mean')
df['Model_encoded'] = df.groupby('Model')['Price'].transform('mean')

# Define numeric features including the one-hot Fuel type columns
fuel_cols = ['Fuel type_Diesel', 'Fuel type_Hybrid', 'Fuel type_Petrol']
numeric_features = ['Engine size', 'Year of manufacture', 'Mileage',
                    'Manufacturer_encoded', 'Model_encoded'] + fuel_cols
#%% PREPARE FEATURES AND TARGET
# Log transforms
df['Engine_log'] = np.log(df['Engine size'] + 1)
df['Mileage_log'] = np.log(df['Mileage'] + 1)

# Features & target
X_np = df[['Engine_log', 'Mileage_log', 'Year of manufacture',
           'Manufacturer_encoded', 'Model_encoded']].values.astype(np.float32)
y_np = df['Price'].values.astype(np.float32).reshape(-1, 1)

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_np)
X = torch.from_numpy(X_scaled.astype(np.float32))

# %% =====================================
# PYTORCH LINEAR REGRESSION
# ========================================
y_mean = y_np.mean()
y_std = y_np.std()
y_scaled = (y_np - y_mean) / y_std
y_true = torch.from_numpy(y_scaled.astype(np.float32))

print("X shape:", X.shape, "y_true shape:", y_true.shape)
print("Features used:", numeric_features)


# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_np)
X = torch.from_numpy(X_scaled.astype(np.float32))


#%% TRAINING LOOP
num_features = X.shape[1]
w = torch.rand(num_features, 1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

num_epochs = 1000
learning_rate = 0.01

for epoch in range(num_epochs):
    y_pred = X @ w + b
    loss_tensor = torch.mean((y_pred - y_true) ** 2)
    loss_tensor.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss_tensor.item()}')

# Predictions (rescale back to original prices)
with torch.no_grad():
    y_pred_scaled = X @ w + b
    y_pred_original = y_pred_scaled * y_std + y_mean  # rescale to original Prices

print('First 5 predicted Prices:', y_pred_original[:5])
print('First 5 actual Prices:', y_np[:5])

#%% CHECK RESULTS AND PLOT
print("Weights:", w.detach().numpy().reshape(-1))
print("Bias:", b.item())

# Visualize predictions vs actual for 'Engine size'
plt.scatter(df['Engine size'], y_np, label='Actual')
plt.scatter(df['Engine size'], y_pred_original.numpy(), label='Predicted', color='red', alpha=0.5)
plt.xlabel('Engine size')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% =====================================
# SCIKIT-LEARN LINEAR REGRESSION
# ========================================
reg = LinearRegression().fit(X_np, y_np)  # y_np shape (n_samples, 1)
print("Coefficients (slopes):", reg.coef_.flatten())
print("Intercept:", reg.intercept_)

#  compare predictions
y_sklearn_pred = reg.predict(X_np)
plt.scatter(df['Engine size'], y_np, label='Actual')
plt.scatter(df['Engine size'], y_sklearn_pred, label='Sklearn Predicted', color='green', alpha=0.5)
plt.xlabel('Engine size')
plt.ylabel('Prices')
plt.legend()
plt.show()


#%% PLOT ACTUAL VS PREDICTED CHARGES (ALL FEATURES)
plt.figure(figsize=(8,6))
plt.scatter(y_np, y_pred_original.numpy(), alpha=0.6)
plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("PyTorch Linear Regression: Actual vs Predicted Prices")
plt.show()

# %% =====================================
# EVALUATION (PyTorch)
# ========================================
from sklearn.metrics import r2_score
r2 = r2_score(y_np, y_pred_original.numpy())
print("R² score (PyTorch model):", r2)


# RESIDUAL PLOT
residuals = y_np - y_pred_original.numpy()  # actual - predicted

plt.figure(figsize=(8,6))
plt.scatter(y_pred_original.numpy(), residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')  # reference line at 0
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot for PyTorch Linear Regression")
plt.show()


