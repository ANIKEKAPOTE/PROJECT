import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load Dataset
@st.cache_data
def load_data():
    file_path = "E:\DS PROJECT\World_development_mesurement (1).csv"  # Update this path
    data = pd.read_csv(file_path)
    return data


# Preprocessing Function
def preprocess_data(data):
    numeric_columns = data.select_dtypes(include=['number']).columns

    # Handle missing values
    data = data.copy()
    for col in numeric_columns:
        data[col].fillna(data[col].mean(), inplace=True)

    # Scale the numeric columns
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numeric_columns])
    return data_scaled, numeric_columns


# Clustering Function
def perform_clustering(data_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    return kmeans, clusters


# Streamlit App
def main():
    st.title("Clustering Application")
    st.write("Analyze global development metrics using clustering.")

    # Load data
    data = load_data()
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Preprocess data
    data_scaled, numeric_columns = preprocess_data(data)

    # User Input: Number of Clusters
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3, step=1)

    # Perform Clustering
    kmeans, clusters = perform_clustering(data_scaled, n_clusters)

    # Add Cluster Labels to Data
    data['Cluster'] = clusters
    st.write("### Clustered Data")
    st.dataframe(data)

    # Visualize Clusters
    st.write("### Cluster Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    st.pyplot(plt)

    # Cluster Distribution
    st.write("### Cluster Distribution")
    st.bar_chart(data['Cluster'].value_counts())


if __name__ == "__main__":
    main()
