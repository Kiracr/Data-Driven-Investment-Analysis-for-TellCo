import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from scripts.engagment_analysis import aggregate_engagement_metrics, find_optimal_k, normalize_metrics, rename_columns

# Load the data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the trained classification model
@st.cache
def load_classification_model(model_path):
    return joblib.load(model_path)

# Load the trained regression model
@st.cache
def load_regression_model(model_path):
    return joblib.load(model_path)

# Function to perform classification
def perform_classification(model, data):
    features = data[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
    return model.predict(features)

# Function to perform regression
def perform_regression(model, data):
    features = data[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
    return model.predict(features)

 
    
def top_10_handsets(df):
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_10_handsets.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Handset Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    return plt.gcf()

def top_3_manufacturers(df):
    top_3_manufacturers = df['Handset Manufacturer'].str.split().str[0].value_counts().head(3)
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    top_3_manufacturers.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), colors=colors, labels=top_3_manufacturers.index, textprops={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Top 3 Handset Manufacturers', fontsize=16, fontweight='bold', color='purple')
    plt.ylabel('')
    plt.axis('equal')
    return plt.gcf()

def top_5_handsets_per_manufacturer(df):
    top_3_manufacturers = df['Handset Manufacturer'].str.split().str[0].value_counts().head(3)
    top_3_df = df[df['Handset Manufacturer'].str.split().str[0].isin(top_3_manufacturers.index)]
    top_5_handsets = top_3_df.groupby(['Handset Manufacturer', 'Handset Type']).size().groupby(level=0, group_keys=False).nlargest(5)
    top_5_handsets = top_5_handsets.reset_index()
    pivot_df = top_5_handsets.pivot(index='Handset Manufacturer', columns='Handset Type', values=0)
    pivot_df.plot(kind='bar', figsize=(12, 8), color=plt.cm.get_cmap('Set3').colors)
    plt.title('Top 5 Handsets per Top 3 Handset Manufacturers', fontsize=16, fontweight='bold')
    plt.xlabel('Handset Manufacturer', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Handset Type', fontsize=12, title_fontsize=12)
    return plt.gcf()

def users_behaviour_on_applications(df):
    aggregated_df = df.groupby('MSISDN/Number').size().reset_index(name='Number of xDR sessions')
    print(aggregated_df)
    
    session_duration_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()
    session_duration_per_user['Session Duration (s)'] = session_duration_per_user['Dur. (ms)'] / 1000
    session_duration_per_user['Session Duration (s)'] = session_duration_per_user['Session Duration (s)'].round(2)
    session_duration_per_user.drop(columns=['Dur. (ms)'], inplace=True)
    aggregated_df = aggregated_df.merge(session_duration_per_user, on='MSISDN/Number', how='left')
    print(aggregated_df)

    total_data_per_user = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
    total_data_per_user['Total DL (MB)'] = total_data_per_user['Total DL (Bytes)'] / (1024 * 1024)
    total_data_per_user['Total DL (MB)'] = total_data_per_user['Total DL (MB)'].round(2)
    total_data_per_user['Total UL (MB)'] = total_data_per_user['Total UL (Bytes)'] / (1024 * 1024)
    total_data_per_user['Total UL (MB)'] = total_data_per_user['Total UL (MB)'].round(2)
    total_data_per_user.drop(columns=['Total DL (Bytes)', 'Total UL (Bytes)'], inplace=True)
    aggregated_df = aggregated_df.merge(total_data_per_user, on='MSISDN/Number', how='left')
    print(aggregated_df)

    total_data_per_app = df.groupby('MSISDN/Number')[['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                                                    'Google DL (Bytes)', 'Google UL (Bytes)',
                                                    'Email DL (Bytes)', 'Email UL (Bytes)',
                                                    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                                                    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                                                    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                                                    'Other DL (Bytes)', 'Other UL (Bytes)']].sum().reset_index()
    aggregated_df = aggregated_df.merge(total_data_per_app, on='MSISDN/Number', how='left')
    print(aggregated_df)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(aggregated_df['MSISDN/Number'], aggregated_df['Number of xDR sessions'], color='blue', marker='o', linestyle='-', linewidth=2, label='Number of xDR sessions')
    ax.plot(aggregated_df['MSISDN/Number'], aggregated_df['Session Duration (s)'], color='green', marker='o', linestyle='-', linewidth=2, label='Session Duration (s)')
    ax.plot(aggregated_df['MSISDN/Number'], aggregated_df['Total DL (MB)'], color='orange', linestyle='--', linewidth=2, label='Total DL (MB)')
    ax.plot(aggregated_df['MSISDN/Number'], aggregated_df['Total UL (MB)'], color='red', linestyle='--', linewidth=2, label='Total UL (MB)')
    ax.set_xlabel('MSISDN/Number', fontsize=14)
    ax.set_ylabel('Volume', fontsize=14)
    ax.set_title('Aggregated User Data', fontsize=16)
    ax.legend(fontsize=12)
    plt.grid(True)
    return plt.gcf()


import seaborn as sns

def visualize_cluster_metrics(cluster_metrics_path):
    # Load the cluster metrics data
    cluster_metrics = pd.read_csv(cluster_metrics_path)

    # Define the metrics to visualize
    metrics = ['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']

    # Initialize a list to store the plots
    plots = []

    # Visualize metrics for each cluster
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Convert data to numeric format if necessary
        cluster_data = pd.to_numeric(cluster_metrics.set_index('Cluster')[metric], errors='coerce')
        
        # Drop NaN values
        cluster_data = cluster_data.dropna()
        
        # Plot the data
        cluster_data.plot(kind='bar', ax=ax, color=['blue', 'green', 'orange'])
        
        ax.set_title(f'{metric} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(metric)
        ax.legend(['Min', 'Max', 'Average', 'Total'], loc='upper right')
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Append the current plot to the list
        plots.append(fig)

    # Return the list of plots as a single figure
    return plt.gcf()

def visualize_top_applications(total_traffic_per_application_path):
    # Load the aggregated total traffic per application data
    total_traffic_per_application = pd.read_csv(total_traffic_per_application_path)

    # Calculate the total traffic for each application
    total_traffic_per_application['Total Traffic'] = total_traffic_per_application.sum(axis=1)

    # Sort the applications by total traffic
    sorted_applications = total_traffic_per_application.sort_values(by='Total Traffic', ascending=False)

    # Plot the top 3 most used applications
    top_3_applications = sorted_applications.head(3)

    # Set style for seaborn
    sns.set(style="whitegrid")

    # Set custom color palette
    custom_palette = sns.color_palette("husl", len(top_3_applications))

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(top_3_applications['Total Traffic'], labels=top_3_applications.index, autopct='%1.1f%%', startangle=140, colors=custom_palette, textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'})
    ax.set_title('Top 3 Most Used Applications', fontsize=14, fontweight='bold', family='Arial')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()

    # Return the plot figure
    return plt.gcf()

def perform_engagement_analysis(cleaned_data_path):
    # Load the data
    df = pd.read_csv(cleaned_data_path)

    # Aggregate engagement metrics per MSISDN
    engagement_metrics = aggregate_engagement_metrics(df)

    # Rename columns for better readability
    engagement_metrics = rename_columns(engagement_metrics)

    # Normalize engagement metrics
    engagement_metrics = normalize_metrics(engagement_metrics)

    # Find optimal value of k
    optimal_k_plot =find_optimal_k(engagement_metrics[['Sessions Frequency', 'Session Duration', 'Total DL Traffic', 'Total UL Traffic']])

    return optimal_k_plot
def main():
    st.set_page_config(page_title="Tellco Streamlit-App", layout="wide")

    # Custom CSS for improved styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            color: #1e3a8a;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
            font-size: 36px;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to", ["Home", "Data", "User Overview Analysis", "Experience Analysis", "Engagement Analysis"])

    if page == "Home":
        st.title('Welcome to Tellco Streamlit-App')
        st.write("This Streamlit app showcases the Tellco data analysis and modeling.")
        st.write("Use the sidebar to navigate through different sections.")

    elif page == "Data":
        st.title('Data')
        # Load and display the data
        df = load_data("results/cleaned_data.csv")
        st.dataframe(df)

    elif page == "User Overview Analysis":
        st.title("User Overview Analysis")
        st.write("This section provides an overview analysis of user behavior, including top handsets used, handset manufacturers, and user engagement metrics.")

        # Load the cleaned data
        df = load_data("results/cleaned_data.csv")

        # Section 1: Top 10 handsets used by the customers
        st.header("Top 10 Handsets Used by Customers")
        st.write("This plot shows the top 10 handsets used by customers based on frequency.")
        top_10_handsets_plot = top_10_handsets(df)
        st.pyplot(top_10_handsets_plot)


       # Section 2: Top 3 handset manufacturers
        st.header("Top 3 Handset Manufacturers")
        st.write("This pie chart illustrates the distribution of top 3 handset manufacturers.")
        top_3_manufacturers_plot = top_3_manufacturers(df)
        st.pyplot(top_3_manufacturers_plot)

        # Section 3: Top 5 handsets per top 3 handset manufacturer
        st.header("Top 5 Handsets per Top 3 Handset Manufacturers")
        st.write("This bar chart displays the top 5 handsets for each of the top 3 handset manufacturers.")
        top_5_handsets_per_manufacturer_plot = top_5_handsets_per_manufacturer(df)
        st.pyplot(top_5_handsets_per_manufacturer_plot)

        # Section 4: Users’ behaviour on applications
        st.header("Users’ Behaviour on Applications")
        st.write("This section presents aggregated user data including the number of xDR sessions, session duration, and total data volume.")
        users_behaviour_on_applications_plot = users_behaviour_on_applications(df)
        st.pyplot(users_behaviour_on_applications_plot)


    elif page == "Engagement Analysis":
        st.title('Engagement Analysis')

        # Section 1: Visualize cluster metrics
        st.header("Cluster Metrics Visualization")
        st.write("This section visualizes metrics for each cluster.")

        # Call the function to visualize cluster metrics and get the plot figure
        cluster_metrics_plot = visualize_cluster_metrics("results/engagement_cluster_stats.csv")

        # Display the plot figure using Streamlit
        st.pyplot(cluster_metrics_plot)

        # Section 2: Visualize top applications
        st.header("Top Applications Visualization")
        st.write("This section visualizes the top 3 most used applications.")

        # Call the function to visualize top applications and get the plot figure
        top_applications_plot = visualize_top_applications("results/total_traffic_per_application.csv")

        # Display the plot figure using Streamlit
        st.pyplot(top_applications_plot)
        
        st.header("Engagement Analysis")
        st.write("This section performs engagement analysis.")

        # Call the function to perform engagement analysis
        optimal_k_plot = perform_engagement_analysis("results/cleaned_data.csv")

        # Display the plot figure using Streamlit
        st.pyplot(optimal_k_plot)

        st.write("Engagement analysis completed.")




if __name__ == "__main__":
    main()
