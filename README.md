# Used-Car-Price-Prediction
A machine learning model that predicts the price of used cars based on features like age, mileage, brand, and condition to help buyers and sellers make informed decisions.
# Car Price Prediction with Linear Regression (PyTorch & Scikit-Learn)

## 📌 Project Overview
This project applies **linear regression** to predict car prices using the [Car Sales Dataset](https://www.kaggle.com/datasets/gagandeep16/car-sales).  
It compares two implementations:
- **PyTorch**: manual gradient descent
- **Scikit-Learn**: built-in linear regression

The goal is to understand how preprocessing, feature engineering, and model implementation affect performance.

---

## 📊 Dataset
The dataset contains information about cars, including:
- **Numerical features**: Engine size, Mileage, Year of manufacture, Price
- **Categorical features**: Fuel type, Manufacturer, Model

Link: [Kaggle – Car Sales Dataset](https://www.kaggle.com/datasets/gagandeep16/car-sales)

---

## ⚙️ Methods

### 🔹 Data Cleaning
- Removed duplicates
- Checked for missing values
- Detected outliers (via IQR and boxplots)

### 🔹 Feature Engineering
- **Log transform**: Engine size, Mileage  
- **One-hot encoding**: Fuel type  
- **Target encoding**: Manufacturer, Model  

### 🔹 Models
1. **PyTorch implementation**  
   - Manual linear regression with gradient descent  
   - Custom training loop  
2. **Scikit-Learn implementation**  
   - `LinearRegression()` model  

### 🔹 Evaluation
- R² score (coefficient of determination)  
- Residual analysis  
- Predicted vs. Actual plots  

---

## 📈 Results

| Model         | R² Score |
|---------------|----------|
| PyTorch       | ~0.76    |
| Scikit-Learn  | ~0.77    |


