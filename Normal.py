import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import Counter
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle


warnings.filterwarnings('ignore')
data = pd.read_excel(r'Employee_Performance.xls')
pd.set_option('display.max_columns', None)
print(data.isnull().sum())
categorical_features = list(data.select_dtypes('object'))
data.Gender = data.Gender.map({'Male': 1, 'Female': 0})
education_map = {
    'Life Sciences': 5,
    'Medical': 4,
    'Marketing': 3,
    'Technical Degree': 2,
    'Other': 1,
    'Human Resources': 0
}
data.EducationBackground = data.EducationBackground.map(education_map)
data.MaritalStatus = data.MaritalStatus.map({'Married': 2, 'Single': 1, 'Divorced': 0})
department_map = {
    'Sales': 5,
    'Development': 4,
    'Research & Development': 3,
    'Human Resources': 2,
    'Finance': 1,
    'Data Science': 0
}
data.EmpDepartment = data.EmpDepartment.map(department_map)
job_role_map = {
    'Sales Executive': 18,
    'Developer': 17,
    'Manager R&D': 16,
    'Research Scientist': 15,
    'Sales Representative': 14,
    'Laboratory Technician': 13,
    'Senior Developer': 12,
    'Manager': 11,
    'Finance Manager': 10,
    'Human Resources': 9,
    'Technical Lead': 8,
    'Manufacturing Director': 7,
    'Healthcare Representative': 6,
    'Data Scientist': 5,
    'Research Director': 4,
    'Business Analyst': 3,
    'Senior Manager R&D': 2,
    'Delivery Manager': 1,
    'Technical Architect': 0
}
data.EmpJobRole = data.EmpJobRole.map(job_role_map)
data.BusinessTravelFrequency = data.BusinessTravelFrequency.map({'Travel_Rarely': 2, 'Travel_Frequently': 1, 'Non-Travel': 0})
data['OverTime'] = data['OverTime'].map({'No': 1, 'Yes': 0})
data['Attrition'] = data['Attrition'].map({'No': 1, 'Yes': 0})
iqr = stats.iqr(data['TotalWorkExperienceInYears'], interpolation='midpoint')
print("IQR:", iqr)
def plot_data(data, feature):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    data[feature].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], dist='norm', plot=pylab)
    plt.show()
plot_data(data, 'YearsSinceLastPromotion')
data['square_YearsSinceLastPromotion'] = data['YearsSinceLastPromotion'] ** (1/2)
plot_data(data, 'square_YearsSinceLastPromotion')
scaler = StandardScaler()  
data[['Age', 'DistanceFromHome', 'EmpHourlyRate', 'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears',
      'TrainingTimesLastYear', 'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 'YearsWithCurrManager',
      'square_YearsSinceLastPromotion']] = scaler.fit_transform(data[['Age', 'DistanceFromHome', 'EmpHourlyRate',
      'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'ExperienceYearsAtThisCompany',
      'ExperienceYearsInCurrentRole', 'YearsWithCurrManager', 'square_YearsSinceLastPromotion']])
data.drop(['EmpNumber', 'YearsSinceLastPromotion'], axis=1, inplace=True)
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, cmap='BuPu')
plt.show()
pd.set_option('display.max_rows', None)
corrmat = data.corr().abs().unstack().sort_values(ascending=False)
corrmat = corrmat[corrmat >= 0.9]
corrmat = corrmat[corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['feature1', 'feature2', 'corr']
print(corrmat)
pca = PCA()
principal_components = pca.fit_transform(data)
pca = PCA(n_components=10)
principal_components = pca.fit_transform(data.drop('PerformanceRating', axis=1))
principle_df = pd.DataFrame(data=principal_components, columns=[f'pca{i+1}' for i in range(10)])
principle_df['PerformanceRating'] = data['PerformanceRating']
principle_df.to_csv('employee_performance_preprocessed_data.csv', index=False)
data = pd.read_csv('employee_performance_preprocessed_data.csv')
X = data.iloc[:, :-1]
y = data['PerformanceRating']
X_train_ols = sm.add_constant(X)
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X_train_ols, y) 
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.20, random_state=42)
ols_model = sm.OLS(y_train, X_train)
ols_results = ols_model.fit()
print(ols_results.summary())
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_train_predict = log_reg.predict(X_train)
log_reg_test_predict = log_reg.predict(X_test)
print("Training accuracy of Logistic Regression model:", accuracy_score(y_train, log_reg_train_predict) * 100)
print("Logistic Regression Classification report (Training):\n", classification_report(y_train, log_reg_train_predict))
print("Testing accuracy of Logistic Regression model:", accuracy_score(y_test, log_reg_test_predict) * 100)
print("Logistic Regression Classification report (Testing):\n", classification_report(y_test, log_reg_test_predict))
multi_reg = sm.OLS(y_train, X_train)
multi_results = multi_reg.fit()
print(multi_results.summary())
t_statistic, p_value = stats.ttest_ind(y_train, y_test)
print(f"T-statistic: {t_statistic}, p-value: {p_value}")
f_statistic, p_value = stats.f_oneway(y_train, y_test)
print(f"F-statistic: {f_statistic}, p-value: {p_value}")
data['PerformanceRating_Binary'] = np.where(data['PerformanceRating'] >= 4, 1, 0)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_train)
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_pca_2, y_train)
x_min, x_max = X_pca_2[:, 0].min() - 1, X_pca_2[:, 0].max() + 1
y_min, y_max = X_pca_2[:, 1].min() - 1, X_pca_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = log_reg_pca.predict(grid).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y_train, s=20, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary (Logistic Regression with PCA)')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_train, ols_results.predict(X_train))
plt.plot(y_train, y_train, color='red')
plt.title('Actual vs Predicted (OLS)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_train, multi_results.predict(X_train))
plt.plot(y_train, y_train, color='red')
plt.title('Actual vs Predicted (Multivariate Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(p_value, 'o', color='blue')
plt.axhline(y=0.05, color='red', linestyle='--')
plt.title("F-test p-values")
plt.xlabel("Features")
plt.ylabel("p-value")
plt.show()
