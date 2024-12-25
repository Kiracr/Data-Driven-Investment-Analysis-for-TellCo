import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator


def load_data(file_path):
    """Load the cleaned data from a CSV file."""
    return pd.read_csv(file_path)

def aggregate_engagement_metrics(df):
    """Aggregate engagement metrics per MSISDN."""
    return df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',          # Sessions frequency
        'Dur. (ms)': 'sum',            # Session duration
        'Total DL (Bytes)': 'sum',     # Total download traffic
        'Total UL (Bytes)': 'sum'      # Total upload traffic
    }).reset_index()

def rename_columns(df):
    """Rename columns for better readability."""
    df.columns = ['MSISDN', 'Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']
    return df

def normalize_metrics(df):
    """Normalize engagement metrics."""
    scaler = MinMaxScaler()
    df[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']] = \
        scaler.fit_transform(df[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']])
    return df

def run_kmeans_clustering(df, n_clusters=3):
    """Run K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']])
    return df

def print_top_10_customers_per_cluster(df):
    """Print the top 10 customers per cluster."""
    for cluster_id in range(df['Cluster'].nunique()):
        top_10_customers = df[df['Cluster'] == cluster_id].nlargest(10, 'Sessions Frequency')
        print(f"Top 10 customers in cluster {cluster_id}:")
        print(top_10_customers)
        print("\n")
        top_10_customers.to_csv(f"results/customers_cluster_{cluster_id}.csv", index=False)

def concatenate_cluster_dataframes(cluster_files, output_file):
    """Concatenate CSV files for each cluster and save to a single CSV file."""
    clusters = []
    for file in cluster_files:
        cluster = pd.read_csv(file)
        clusters.append(cluster)
    
    all_customers = pd.concat(clusters, ignore_index=True)
    all_customers.to_csv(output_file, index=False)

def compute_cluster_metrics(df):
    """Compute non-normalized metrics for each cluster."""
    cluster_metrics = df.groupby('Cluster').agg({
        'Sessions Frequency': ['min', 'max', 'mean', 'sum'],
        'Session Duration': ['min', 'max', 'mean', 'sum'],
        'Total DL Traffic': ['min', 'max', 'mean', 'sum'],
        'Total UL Traffic': ['min', 'max', 'mean', 'sum']
    }).reset_index()
    return cluster_metrics

def save_cluster_metrics_to_csv(df, output_file):
    """Save cluster metrics to a CSV file."""
    df.to_csv(output_file, index=False)




def aggregate_traffic_per_application(df):
    """Aggregate user total traffic per application."""
    # List of columns representing different applications
    application_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                           'Google DL (Bytes)', 'Google UL (Bytes)',
                           'Email DL (Bytes)', 'Email UL (Bytes)',
                           'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                           'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                           'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                           'Other DL (Bytes)', 'Other UL (Bytes)']
    
    # Sum the traffic for each application
    df['Total Social Media Traffic (Bytes)'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
    df['Total Google Traffic (Bytes)'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
    df['Total Email Traffic (Bytes)'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)']
    df['Total Youtube Traffic (Bytes)'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
    df['Total Netflix Traffic (Bytes)'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
    df['Total Gaming Traffic (Bytes)'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
    df['Total Other Traffic (Bytes)'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']
    
    # Aggregate total traffic per application for each user
    total_traffic_per_application = df.groupby('MSISDN/Number').agg({
        'Total Social Media Traffic (Bytes)': 'sum',
        'Total Google Traffic (Bytes)': 'sum',
        'Total Email Traffic (Bytes)': 'sum',
        'Total Youtube Traffic (Bytes)': 'sum',
        'Total Netflix Traffic (Bytes)': 'sum',
        'Total Gaming Traffic (Bytes)': 'sum',
        'Total Other Traffic (Bytes)': 'sum'
    }).reset_index()
    
    return total_traffic_per_application

def top_10_engaged_users_per_application(df, application_column):
    """Derive the top 10 most engaged users per application."""
    top_10_users = df.nlargest(10, application_column)
    return top_10_users


def find_optimal_k(df):
    """Find the optimal value of k using the elbow method."""
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters', fontsize=12, fontweight='bold', family='Arial')
    plt.ylabel('Distortion', fontsize=12, fontweight='bold', family='Arial')
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold', family='Arial')
    plt.xticks(K, fontsize=10, fontweight='bold', family='Arial')
    plt.yticks(fontsize=10, fontweight='bold', family='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Find the optimal k
    kl = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    optimal_k = kl.elbow

    print(f"Optimal value of k: {optimal_k}")

# def create_kmeans_model(df, n_clusters=3):
#     """Create and train a K-Means model."""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df['Cluster'] = kmeans.fit_predict(df[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']])
#     return kmeans, df
def train_regression_model(df):
    # Split data into features (X) and target variable (y)
    X = df[['Session Duration', 'Total DL Traffic', 'Total UL Traffic']]
    y = df['Sessions Frequency']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model
def main():
    # Load the cleaned data
    df = load_data("results/cleaned_data.csv")

    # Aggregate engagement metrics per MSISDN
    engagement_metrics = aggregate_engagement_metrics(df)

    # Rename columns for better readability
    engagement_metrics = rename_columns(engagement_metrics)

    # Normalize engagement metrics
    engagement_metrics = normalize_metrics(engagement_metrics)

    # kmeans_model, clustered_data = create_kmeans_model(engagement_metrics)

    # Run K-Means clustering (k=3)
    engagement_metrics = run_kmeans_clustering(engagement_metrics)

    # Print the top 10 customers per cluster
    print_top_10_customers_per_cluster(engagement_metrics)

    total_traffic_per_application = aggregate_traffic_per_application(df)

    total_traffic_per_application.to_csv("results/total_traffic_per_application.csv")


    # Derive the top 10 most engaged users per application (example for Social Media)
    top_10_social_media_users = top_10_engaged_users_per_application(total_traffic_per_application, 'Total Social Media Traffic (Bytes)')
    print("Top 10 Most Engaged Users in Social Media:")
    print(top_10_social_media_users)

    # Concatenate CSV files for each cluster
    cluster_files = ["results/customers_cluster_0.csv", "results/customers_cluster_1.csv", "results/customers_cluster_2.csv"]
    concatenate_cluster_dataframes(cluster_files, "results/all_customers_clusters.csv")

    # Compute non-normalized metrics for each cluster
    cluster_metrics = compute_cluster_metrics(engagement_metrics)

    # Save cluster metrics to CSV
    save_cluster_metrics_to_csv(cluster_metrics, "results/engagement_cluster_stats.csv")

    find_optimal_k(engagement_metrics[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']])
    # joblib.dump(kmeans_model, "models/kmeans_model.pkl")
    joblib.dump(train_regression_model, 'models/regression_model.pkl')

if __name__ == "__main__":
    main()
