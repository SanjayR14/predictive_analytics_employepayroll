import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.multivariate.manova import MANOVA
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm


# Read the dataset
data = pd.read_csv("C:/Users/adity/Documents/PA Project 2024/Employee dataset csv.csv")

# Select numeric columns
numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE', 'TRAINING HOURS',
                   'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE', 'PERFORMANCE RATINGS']

data_numeric = data[numeric_columns].copy()

# Drop rows with missing values
data_numeric.dropna(inplace=True)

# Define features and target variable
X = data_numeric.drop(columns=['PERFORMANCE RATINGS'])  # Features
y = data_numeric['PERFORMANCE RATINGS']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Add a constant term to the features for OLS regression
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

# Fit the OLS model
ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print OLS regression results
print(ols_results.summary())

# Predict using the OLS model
y_pred_ols = ols_results.predict(X_test_ols)

# Calculate metrics for OLS model
mse_ols = mean_squared_error(y_test, y_pred_ols)
rmse_ols = np.sqrt(mse_ols)
r2_ols = r2_score(y_test, y_pred_ols)

# Print metrics for OLS model
print("\nResults for Ordinary Least Squares (OLS) Regression Model:")
print("Mean Squared Error:", mse_ols)
print("Root Mean Squared Error:", rmse_ols)
print("R-squared:", r2_ols)

# Residual Analysis for OLS model
residuals_ols = y_test - y_pred_ols
plt.figure("Residual Analysis (OLS)", figsize=(10, 6))
plt.scatter(y_pred_ols, residuals_ols, color='blue')
plt.xlabel('Predicted Performance Ratings (OLS)')
plt.ylabel('Residuals (OLS)')
plt.title('Residual Analysis (OLS)')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)
plt.show()

# Q-Q Plot for OLS model
fig = plt.figure("Normal Q-Q Plot (OLS)", figsize=(10, 6))
ax = fig.add_subplot(111)
qqplot(residuals_ols, line='s', ax=ax)
ax.set_title('Normal Q-Q Plot (OLS)')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
ax.grid(True)
plt.show()


# Multiple Regression model
def multiple_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, rmse, r2, model

y_pred_multiple, mse_multiple, rmse_multiple, r2_multiple, model_multiple = multiple_linear_regression(X_train, X_test, y_train, y_test)

# Perform MANOVA
manova = MANOVA(X_train, y_train)

print("\nResults for Multiple Linear Regression Model:")
print("Mean Squared Error:", mse_multiple)
print("Root Mean Squared Error:", rmse_multiple)
print("R-squared:", r2_multiple)

print("\nMANOVA Results:")
print(manova.mv_test())


# Visualization: Correlation Matrix
plt.figure("Correlation Matrix", figsize=(12, 6))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# Visualization: Actual vs Predicted Performance Ratings
plt.figure("Actual vs Predicted Performance Ratings", figsize=(10, 6))
plt.scatter(y_test, y_pred_multiple, color='blue')
plt.xlabel('Actual Performance Ratings')
plt.ylabel('Predicted Performance Ratings (Multiple Regression)')
plt.title('Actual vs Predicted Performance Ratings (Multiple Regression)')
plt.grid(True)
plt.show()
