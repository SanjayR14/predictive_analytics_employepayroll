import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("C:/Users/adity/Documents/PA Project 2024/Employee dataset csv.csv")

numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE', 'TRAINING HOURS',
                   'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE', 'PERFORMANCE RATINGS']


data_numeric = data[numeric_columns].copy()
''' Avoids warning '''

data_numeric.dropna(inplace=True)
''' Handles missing values if any ''' 


X = data_numeric.drop(columns=['PERFORMANCE RATINGS'])  # Features
y = data_numeric['PERFORMANCE RATINGS']  # Target variable


'''  Split the data into training and testing sets (80% train, 20% test) 
 80/100 = 0.8, 20/100 = 0.2 and random state is for data splitting    '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Predicted Ratings: \n" ,y_pred)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


plt.figure(figsize=(12, 6))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Performance Ratings')
plt.ylabel('Predicted Performance Ratings')
plt.title('Actual vs Predicted Performance Ratings')
plt.show()
