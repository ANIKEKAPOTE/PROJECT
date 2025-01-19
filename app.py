import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.impute import SimpleImputer

print(sns.__version__)

import subprocess
import sys

# Ensure seaborn is installed
try:
    import seaborn as sns
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

WD = pd.read_csv("E:\DS PROJECT\World_development_mesurement (1).csv")
WD

WD.rename(columns={'GDP#':'GDP', 'Country?':'Country'}, inplace=True)
WD.set_index('GDP',inplace=True)
WD

WD.describe()

WD.isnull().sum()
for feature in WD.columns:
    airline_data1 = WD.copy()
    airline_data1[feature].hist(bins=25)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()
WD.columns

outlier = WD.copy()

# Check for numeric data
numeric_columns = [
    'Birth Rate', 'CO2 Emissions', 'Days to Start Business',
    'Population Total', 'Tourism Inbound', 'Tourism Outbound'
]

# Check if columns exist and are numeric
for col in numeric_columns:
    if col not in outlier.columns or not pd.api.types.is_numeric_dtype(outlier[col]):
        print(f"Warning: Column '{col}' is either missing or not numeric.")
        outlier[col] = pd.to_numeric(outlier[col], errors='coerce')  # Convert to numeric if possible

fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=False, sharey=False)
# Numeric variables (horizontal orientation)
sns.boxplot(x='Birth Rate', data=outlier, palette='crest', ax=axes[0])
sns.boxplot(x='CO2 Emissions', data=outlier, palette='crest', ax=axes[1])
sns.boxplot(x='Days to Start Business', data=outlier, palette='crest', ax=axes[2])
sns.boxplot(x='Population Total', data=outlier, palette='crest', ax=axes[3])
sns.boxplot(x='Tourism Inbound', data=outlier, palette='crest', ax=axes[4])
sns.boxplot(x='Tourism Outbound', data=outlier, palette='crest', ax=axes[5])
plt.tight_layout(pad=2.0)
plt.show()

Balance = WD[['Birth Rate','Country']].sort_values('Country', ascending = False)
ax = sns.barplot(x='Birth Rate', y='Country', data= Balance)
ax.set(xlabel = 'Birth Rate', ylabel= 'Country')
plt.title('Country : Number of miles eligible for Birth Rate travel')
plt.xticks(rotation=90)
plt.show()

corr_matrix = WD.corr()
corr_matrix["Birth Rate"].sort_values(ascending=False)

f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(WD.corr(), annot=True, linewidths =.5, fmt ='.1f',ax=ax)
plt.show()

# Copy the DataFrame
outlier = WD.copy()

# Convert percentage strings to numeric values
for col in outlier.columns:
    if outlier[col].dtype == 'object':
        # Check for percentage strings
        if outlier[col].str.contains('%', na=False).any():
            outlier[col] = outlier[col].str.replace('%', '').astype(float) / 100
        else:
            # Handle non-numeric columns (e.g., drop them or encode them)
            outlier[col] = pd.to_numeric(outlier[col], errors='coerce')

# Fill missing values with column mean (or other suitable method)
outlier.fillna(outlier.mean(), inplace=True)
# Apply StandardScaler
standard_scaler = StandardScaler()
std_airline = standard_scaler.fit_transform(outlier)

# Check the shape of the scaled data
print("Shape of scaled data:", std_airline.shape)
# Convert percentage strings to numeric values
for col in outlier.columns:
    if outlier[col].dtype == 'object':
        # Check for percentage strings
        if outlier[col].str.contains('%', na=False).any():
            outlier[col] = outlier[col].str.replace('%', '').astype(float) / 100
        else:
            # Handle non-numeric columns (drop or encode as needed)
            outlier[col] = pd.to_numeric(outlier[col], errors='coerce')

# Fill missing values with the column mean
outlier.fillna(outlier.mean(), inplace=True)

# Apply MinMaxScaler
minmax = MinMaxScaler()
norm_airline = minmax.fit_transform(outlier)

# Check the shape of the normalized data
print("Shape of normalized data:", norm_airline.shape)

# Check the shape of the normalized data
print("Shape of normalized data:", norm_airline.shape)

# Impute missing values for numeric columns
numeric_columns = WD.select_dtypes(include=['number']).columns
imputer_num = SimpleImputer(strategy='mean')
WD[numeric_columns] = imputer_num.fit_transform(WD[numeric_columns])

# Normalize/Scale the numeric data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(WD[numeric_columns])

# Fit K-Means model with a specified number of clusters (e.g., 3 clusters)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
WD['Cluster'] = kmeans.fit_predict(scaled_data)

# Add cluster labels to the DataFrame
print(WD.head())

# Plot the clusters (if data is 2D or reduced to 2D using PCA or t-SNE)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=WD['Cluster'], cmap='viridis')
plt.title(f'K-Means Clustering with {n_clusters} Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

data = {
    'GDP': [50000, 20000, 30000, 10000, 15000],
    'Life Expectancy': [80, 70, 75, 60, 65],
    'Literacy Rate': [0.99, 0.85, 0.92, 0.75, 0.80],
    'Country': ['A', 'B', 'C', 'D', 'E']
}
df = pd.DataFrame(data)

# Create a pairplot
sns.pairplot(df, diag_kind='kde', corner=True)

# Show the plot
plt.show()
# Evaluate using inertia (within-cluster sum of squares)
print(f'Inertia (Within-cluster sum of squares): {kmeans.inertia_}')
# You can also try to find the optimal number of clusters using the Elbow method
from sklearn.cluster import KMeans

inertia_values = []
n_clusters_range = range(1, 11)

for n in n_clusters_range:
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Method to determine optimal clusters
plt.plot(n_clusters_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Compute silhouette score for K-Means
kmeans_silhouette = silhouette_score(scaled_data, WD['Cluster'])
print(f'Silhouette Score for K-Means: {kmeans_silhouette}')
