import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def perform_clustering(df):
    # Selecting the numerical columns for clustering
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    # Dropping NaN values and selecting the first 10,000 rows for faster processing
    numeric_cols = numeric_cols.dropna()
    numeric_cols = numeric_cols.head(10000)

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_cols)

    # Performing K-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Assigning the cluster labels to a new column in the DataFrame
    clustered_df = df.loc[numeric_cols.index].copy()
    clustered_df['Cluster'] = cluster_labels

    return clustered_df


def perform_classification(df):
    features = df[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]

    target = df['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    return clf

def aggregate_information(df):
    grouped_handset_types = df.groupby('MSISDN/Number')['Handset Type']
    handset_type = grouped_handset_types.agg(lambda x: x.value_counts().index[0] if not x.empty and x.value_counts().any() else 'Unknown')
    avg_tcp_retransmission = df.groupby('MSISDN/Number')['TCP DL Retrans. Vol (Bytes)'].mean()
    average_rtt = df.groupby('MSISDN/Number')['Avg RTT DL (ms)'].mean()
    average_throughput = df.groupby('MSISDN/Number')['Avg Bearer TP DL (kbps)'].mean()
    return avg_tcp_retransmission, average_rtt, handset_type, average_throughput

def compute_list_values(df):
    top_10_tcp_values = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
    bottom_10_tcp_values = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
    most_frequent_tcp_values = df['TCP DL Retrans. Vol (Bytes)'].value_counts().head(10)
    top_10_rtt_values = df['Avg RTT DL (ms)'].nlargest(10)
    bottom_10_rtt_values = df['Avg RTT DL (ms)'].nsmallest(10)
    most_frequent_rtt_values = df['Avg RTT DL (ms)'].value_counts().head(10)
    top_10_throughput_values = df['Avg Bearer TP DL (kbps)'].nlargest(10)
    bottom_10_throughput_values = df['Avg Bearer TP DL (kbps)'].nsmallest(10)
    most_frequent_throughput_values = df['Avg Bearer TP DL (kbps)'].value_counts().head(10)
    return (top_10_tcp_values, bottom_10_tcp_values, most_frequent_tcp_values,
            top_10_rtt_values, bottom_10_rtt_values, most_frequent_rtt_values,
            top_10_throughput_values, bottom_10_throughput_values, most_frequent_throughput_values)

def compute_report(df):
    avg_throughput_per_handset_type = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
    avg_tcp_retransmission_per_handset_type = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
    return avg_throughput_per_handset_type, avg_tcp_retransmission_per_handset_type

def main():
    df = pd.read_csv("results/cleaned_data.csv")
    clustered_df = perform_clustering(df)

    avg_tcp_retransmission, avg_rtt, handset_type, avg_throughput = aggregate_information(clustered_df)
    
    top_10_tcp_values, bottom_10_tcp_values, most_frequent_tcp_values, \
    top_10_rtt_values, bottom_10_rtt_values, most_frequent_rtt_values, \
    top_10_throughput_values, bottom_10_throughput_values, most_frequent_throughput_values = compute_list_values(clustered_df)
    
    avg_throughput_per_handset_type, avg_tcp_retransmission_per_handset_type = compute_report(clustered_df)
    clf = perform_classification(clustered_df)

    joblib.dump(clf, 'models/classification_model.pkl')
    clustered_df.to_csv("results/experience_clusters.csv", index=False)


if __name__ == "__main__":
    main()
