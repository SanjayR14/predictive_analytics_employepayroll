import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Set up Seaborn style
sns.set_style('darkgrid')

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_excel(r'Employee_Performance.xls')

# Display all columns
pd.set_option('display.max_columns', None)

''' # Display first and last 5 rows, and dataset info
print(data.head())
print(data.tail())
print(data.shape)
print(data.size)
print(data.info()) 

# Display column names and descriptive statistics
print(data.columns) '''

print("\n STATISTICAL ANALYSIS OF NUMERICAL FEATURES: \n", data.describe().T)
print("\n STATISTICAL ANALYSIS OF CATEGORICAL FEATURES: \n", data.describe(include='O').T)

'''
# Plot histogram for a single continuous feature
def plot_histogram(column_name, data):
    plt.figure(figsize=(10, 7))
    sns.histplot(x=column_name, data=data)
    plt.xlabel(column_name, fontsize=20)
    plt.show()

plot_histogram('Age', data)

# Plot count plot for a single categorical feature
def plot_countplot(column_name, data):
    plt.figure(figsize=(10, 7))
    sns.countplot(x=column_name, data=data)
    plt.xlabel(column_name, fontsize=20)
    plt.show()

plot_countplot('Gender', data)

# Plot line plot for a single relationship between features
def plot_lineplot(x, y, data):
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=x, y=y, data=data)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.show()

plot_lineplot('Age', 'TotalWorkExperienceInYears', data)

# Plot line plot for 'Age' vs 'TotalWorkExperienceInYears' colored by 'PerformanceRating'
plt.figure(figsize=(20, 10))
sns.lineplot(x='Age', y='TotalWorkExperienceInYears', hue='PerformanceRating', data=data)
plt.xlabel('Age', fontsize=20)
plt.ylabel('TotalWorkExperienceInYears', fontsize=15)
plt.show()''' 

# 3D clustering
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Select features for clustering
features = ['Age', 'TotalWorkExperienceInYears', 'EmpHourlyRate']

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=3)
kmeans.fit(data[features])
labels = kmeans.labels_


# Plot 3D scatter plot
scatter = ax.scatter(data['Age'], data['TotalWorkExperienceInYears'], data['EmpHourlyRate'], c=labels, cmap='viridis')

# Add legend
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Clusters")
ax.add_artist(legend1)

# Set labels and title
ax.set_xlabel('Age')
ax.set_ylabel('TotalWorkExperienceInYears')
ax.set_zlabel('EmpHourlyRate')
plt.title('3D Clustering')
plt.show()
