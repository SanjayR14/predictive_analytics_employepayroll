import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set up Seaborn style
sns.set_style('darkgrid')

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_excel(r'Employee_Performance.xls')

# Display all columns
pd.set_option('display.max_columns', None)

# Display first and last 5 rows, and dataset info
print(data.head())
print(data.tail())
print(data.shape)
print(data.size)
print(data.info())

# Display column names and descriptive statistics
print(data.columns)
print(data.describe().T)
print(data.describe(include='O').T)

# Plot histograms for continuous features
def plot_histogram(column_name, data):
    plt.figure(figsize=(10, 7))
    sns.histplot(x=column_name, data=data)
    plt.xlabel(column_name, fontsize=20)
    plt.show()

plot_histogram('Age', data)
plot_histogram('EmpHourlyRate', data)
plot_histogram('TotalWorkExperienceInYears', data)
plot_histogram('ExperienceYearsAtThisCompany', data)

# Plot count plots for categorical features
categorical_features = ['Gender', 'EducationBackground', 'MaritalStatus', 'BusinessTravelFrequency', 'DistanceFromHome',
                        'EmpEducationLevel', 'EmpEnvironmentSatisfaction', 'EmpJobInvolvement', 'EmpJobLevel',
                        'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime']

plt.figure(figsize=(20, 25))
plot_no = 1
for column in categorical_features:
    if plot_no <= 13:
        plt.subplot(4, 3, plot_no)
        sns.countplot(x=column, data=data)
        plt.xlabel(column, fontsize=20)
    plot_no += 1
plt.tight_layout()
plt.show()

# Plot count plots for other features
other_features = ['EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction', 'TrainingTimesLastYear', 'EmpWorkLifeBalance',
                  'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Attrition', 
                  'PerformanceRating']

plt.figure(figsize=(20, 22))
plot_no = 1
for column in other_features:
    if plot_no <= 10:
        plt.subplot(3, 3, plot_no)
        sns.countplot(x=column, data=data)
        plt.xlabel(column, fontsize=20)
    plot_no += 1
plt.tight_layout()
plt.show()

# Plot count plot for 'EmpDepartment'
plt.figure(figsize=(10, 7))
sns.countplot(x='EmpDepartment', data=data)
plt.xlabel('EmpDepartment', fontsize=20)
plt.show()

# Plot count plot for 'EmpJobRole'
plt.figure(figsize=(20, 10))
sns.countplot(x='EmpJobRole', data=data)
plt.xticks(rotation='vertical')
plt.xlabel('EmpJobRole', fontsize=20)
plt.show()

# Plot line plots for relationships between features
def plot_lineplot(x, y, data):
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=x, y=y, data=data)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.show()

plot_lineplot('Age', 'TotalWorkExperienceInYears', data)
plot_lineplot('ExperienceYearsAtThisCompany', 'TotalWorkExperienceInYears', data)
plot_lineplot('EmpLastSalaryHikePercent', 'NumCompaniesWorked', data)
plot_lineplot('YearsSinceLastPromotion', 'ExperienceYearsInCurrentRole', data)
plot_lineplot('EmpHourlyRate', 'YearsWithCurrManager', data)
plot_lineplot('DistanceFromHome', 'EmpLastSalaryHikePercent', data)

# Plot line plot for 'Age' vs 'TotalWorkExperienceInYears' colored by 'PerformanceRating'
plt.figure(figsize=(20, 10))
sns.lineplot(x='Age', y='TotalWorkExperienceInYears', hue='PerformanceRating', data=data)
plt.xlabel('Age', fontsize=20)
plt.ylabel('TotalWorkExperienceInYears', fontsize=15)
plt.show()
