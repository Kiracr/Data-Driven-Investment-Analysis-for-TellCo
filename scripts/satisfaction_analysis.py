import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

def fill_missing_values(data):
    # Fill missing values in numeric columns with the mean
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    # Fill missing values in datetime columns with the most frequent value if available
    datetime_columns = data.select_dtypes(include='datetime').columns
    if not data[datetime_columns].empty:
        mode_values = data[datetime_columns].mode().iloc[0]
        data[datetime_columns] = data[datetime_columns].fillna(mode_values)
    
    return data

def perform_data_transformation(data):
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=np.number)
    
    # Perform Min-Max scaling on the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    return scaled_data

def assign_engagement_score(user_data, cluster_data):
    distances = euclidean_distances(user_data, cluster_data[:, :-1])  # Exclude the last column from cluster_data
    least_engaged_cluster_index = distances.sum(axis=1).argmin()
    engagement_scores = distances[:, least_engaged_cluster_index]
    return engagement_scores

def assign_experience_score(user_data, cluster_data):
    distances = euclidean_distances(user_data, cluster_data[:, :-1])  # Exclude the last column from cluster_data
    worst_experience_cluster_index = distances.sum(axis=1).argmax()
    experience_scores = distances[:, worst_experience_cluster_index]
    return experience_scores

def main():
    # Read cleaned data directly
    user_data = pd.read_csv("results/cleaned_data.csv")
    
    # Fill missing values with mean
    user_data = fill_missing_values(user_data)
    
    # Perform data transformation
    user_data_transformed = perform_data_transformation(user_data.drop('MSISDN/Number', axis=1))
    
    clustered_data = pd.read_csv("results/experience_clusters.csv")
    numeric_columns = user_data.select_dtypes(include=np.number).columns

    cluster_data = clustered_data.groupby('Cluster')[numeric_columns].mean().values
    
    # Assign engagement scores
    engagement_scores = assign_engagement_score(user_data_transformed, cluster_data)
    
    # Assign experience scores
    experience_scores = assign_experience_score(user_data_transformed, cluster_data)
    
    user_data_with_scores = user_data.copy()
    user_data_with_scores['Engagement Score'] = engagement_scores
    user_data_with_scores['Experience Score'] = experience_scores
    
    user_data_with_scores.to_csv("user_data_with_scores.csv", index=False)

if __name__ == "__main__":
    main()
