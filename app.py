import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a backend compatible with headless environments
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("E:\WDM.csv")

df.head()

# Check dataset dimensions and data types
print("Dataset Dimensions:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Check data types
df.info()


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# List numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Plot histograms for numeric features
df[numeric_columns].hist(figsize=(15, 10), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Numeric Features Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# Convert percentage strings to numeric values where applicable
def clean_percentage(column):
    if column.dtypes == 'object':  # Only process object (string) columns
        try:
            return column.str.replace('%', '').astype(float) / 100
        except:
            return column  # Return the column as is if conversion fails
    return column  # Return numeric columns as is

# Apply the cleaning function selectively
df_cleaned = df.copy()  # Create a copy to avoid modifying the original DataFrame
df_cleaned = df_cleaned.apply(clean_percentage)

# Check for non-numeric columns after cleaning
non_numeric_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Non-numeric Columns:", non_numeric_columns)

# Drop non-numeric columns for correlation analysis
df_numeric = df_cleaned.drop(columns=non_numeric_columns)

# Calculate correlations
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for GDP vs CO2 Emissions
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Birth Rate', y='CO2 Emissions', color='blue')
plt.title('Birth Rate vs CO2 Emissions')
plt.xlabel('country')
plt.ylabel('CO2 Emissions')
plt.grid()
plt.show()

# Percentage of missing data in each column
missing_percentage = (df_cleaned.isnull().sum() / len(df)) * 100
print("Percentage of Missing Data:")
print(missing_percentage[missing_percentage > 0])

# Separate numeric and non-numeric columns
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
non_numeric_columns = df_cleaned.select_dtypes(exclude=['number']).columns

# Fill missing values for numeric columns with their mean
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())

# Fill missing values for non-numeric columns with a placeholder or mode
df_cleaned[non_numeric_columns] = df_cleaned[non_numeric_columns].fillna("Unknown")

# Verify there are no missing values left
print("Missing Values After Imputation:")
print(df_cleaned.isnull().sum())

# Create numeric_df by selecting numeric columns from df
numeric_df = df_cleaned[numeric_columns]

# heatmap of corr > (+ or -) 0.5

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'numeric_df' is already defined from your previous code.
# If not, replace numeric_df with the actual DataFrame containing numeric columns.

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Filter for correlations greater than 0.5 (absolute value)
correlation_matrix = correlation_matrix[abs(correlation_matrix) > 0.8]

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Correlation > 0.5)')
plt.show()

# Create nu_df DataFrame with selected columns from numeric_df
nu_df = numeric_df[['Birth Rate', 'Energy Usage', 'Infant Mortality Rate']]

nu_df.columns

from sklearn.preprocessing import StandardScaler
# Standardization
scaler = StandardScaler()

# Now we can use nu_df for scaling
scaled_df = scaler.fit_transform(nu_df)

nu_df.isnull().sum()

# Pairplot for initial insights
sns.pairplot(data=nu_df.iloc[:, :5])  # Plotting only first 5 columns for simplicity
plt.show()

#K-Means Clustering
# Finding the optimal number of clusters using Elbow Method
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    # Use the correct variable name 'scaled_df' here
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Final K-Means Model
optimal_clusters = 2 # Based on the Elbow plot
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans_model.fit_predict(scaled_df)
import numpy as np

# Get the centroids of the clusters
centroids = kmeans_model.cluster_centers_

# Print the centroids for interpretation
print("Cluster Centroids:")
print(centroids)

if centroids[0][1] > centroids[1][1] and centroids[0][2] > centroids[1][2]:
    cluster_labels = {0: "Developed", 1: "Underdeveloped"}
else:
    cluster_labels = {0: "Underdeveloped", 1: "Developed"}


# Add the cluster labels as a new column 'developed' to the nu_df DataFrame
da = nu_df.copy()  # Create a copy to avoid modifying the original
da['developed'] = kmeans_labels

# Map cluster labels to 0 and 1 based on your earlier defined mapping (cluster_labels)
da['developed'] = da['developed'].map(lambda x: 1 if cluster_labels[x] == 'Developed' else 0)

#Now da dataframe has the new column 'developed' with values 0 and 1.
print(da.head())

# Display the first 5 rows where 'developed' is equal to 1
print(da[da['developed'] == 1].head())

kmeans_labels

len(kmeans_labels)

from sklearn.metrics import silhouette_score, davies_bouldin_score
# K-Means Metrics
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
kmeans_davies = davies_bouldin_score(scaled_df, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"K-Means Davies-Bouldin Score: {kmeans_davies}")

#saving the model
import joblib

joblib.dump(kmeans_model, 'kmeans_model.joblib')

#HIERARCHICAL CLUSTERING (agglomerative approach)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#Hierarchical Clustering
# Dendrogram
linkage_matrix = linkage(scaled_df, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Assigning clusters from Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=2, linkage='ward')
agglomerative_labels = agglomerative.fit_predict(scaled_df)

agglomerative_labels

# prompt: find the silhoutte score of agglomerative

# Calculate and print the silhouette score for agglomerative clustering
agglomerative_silhouette = silhouette_score(scaled_df, agglomerative_labels)
print(f"Agglomerative Silhouette Score: {agglomerative_silhouette}")

#DBSCAN
from sklearn.neighbors import NearestNeighbors
nearest_n = NearestNeighbors(n_neighbors=2)
nearest_n.fit(scaled_df)
distances, indices = nearest_n.kneighbors(scaled_df)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title('K-distance Graph')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon')
plt.show()

#DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(scaled_df)

dbscan_labels

import numpy as np
dbscancluster_labels = np.unique(dbscan_labels)
print(f"Unique values in dbscan_labels: {dbscancluster_labels}")

# DBSCAN Metrics
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(scaled_df, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
else:
    dbscan_silhouette = None
    print("DBSCAN did not form multiple clusters.")

#TUNED DBSCAN MODEL
from sklearn.model_selection import ParameterGrid

# Define the parameter grid for DBSCAN
param_grid = {
    'eps': [0.3, 0.5, 0.7, 1.0],  # Adjust the range of epsilon values
    'min_samples': [1, 2, 3, 5, 10, 15] # Adjust the range of min_samples values
}

best_score = -1
best_params = {}

for params in ParameterGrid(param_grid):
    dbscan_model = DBSCAN(**params)
    Tuned_dbscan_labels = dbscan_model.fit_predict(scaled_df)

    # Handle cases where DBSCAN creates only one cluster
    if len(set(Tuned_dbscan_labels)) > 1:
        score = silhouette_score(scaled_df, Tuned_dbscan_labels)
        if score > best_score:
            best_score = score
            best_params = params

print(f"Best DBSCAN parameters: {best_params}")
print(f"Best Silhouette Score: {best_score}")

# Train final model with best hyperparameters
best_dbscan_model = DBSCAN(**best_params)
best_dbscan_model.fit(scaled_df)

#Comparative Analysis
results = {
    'Model': ['K-Means', 'Hierarchical', 'TUNED DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, agglomerative_silhouette, best_score],
    'Davies-Bouldin Score': [kmeans_davies, None, None]
}
results_df = pd.DataFrame(results)
print("Comparative Analysis of Clustering Models:")
print(results_df)

# Plot the K-means clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the Tuned DBscan clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=Tuned_dbscan_labels, cmap='viridis')
plt.title('Tuned DBscan Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the Hierarchical clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=agglomerative_labels, cmap='viridis')
plt.title('Hierararchical Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

joblib.dump(scaler, 'scaler.joblib')

#Deployment
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained KMeans model and scaler
kmeans_model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')  # Load the pre-trained scaler

st.title('Country Development Classification')

# Input fields in the Streamlit app
Birth_Rate = st.number_input("Enter Birth Rate", value=0.0)
Energy_Usage = st.number_input("Enter Energy Usage",  value=0.0)
Infant_Mortality_Rate = st.number_input("Enter Infant Mortality Rate", value=0.0)

# Create a dictionary for user input
input_data = {
    'Birth Rate': Birth_Rate,
    'Energy Usage': Energy_Usage,
    'Infant Mortality Rate': Infant_Mortality_Rate
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Cluster labels mapping
cluster_labels = {0: "Underdeveloped", 1: "Developed"}

# Validate and process input
if input_df.isnull().values.any():
    st.error("Please fill in all the required fields.")
else:
    try:
        # Standardize input data using the loaded scaler
        scaled_input_data = scaler.transform(input_df)
        # Predict the cluster
        cluster_id = kmeans_model.predict(scaled_input_data)[0]
        # Get the human-readable label
        cluster_label = cluster_labels[cluster_id]
        # Display the result
        st.success(f'The country is classified as: {cluster_label}')
    except Exception as e:
        st.error(f"An error occurred: {e}")