import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load environment variables
load_dotenv()

# Access the environment variables
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

# Create the database connection string
db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create the SQLAlchemy engine
engine = create_engine(db_url)

# Load data from PostgreSQL
query = """
SELECT * FROM public.xdr_data
"""
df = pd.read_sql(query, engine)

# Handle missing values (replace with mean for numeric and mode for categorical)
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype == 'float64' else col.fillna(col.mode()[0]))

# Handle outliers (replace with mean)
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), col] = df[col].mean()

# User Overview Analysis
def get_user_overview():
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

    fig_top_10_handsets = px.bar(x=top_10_handsets.values, y=top_10_handsets.index, title='Top 10 Handsets Used by Customers', labels={'x': 'Number of Users', 'y': 'Handset'})
    fig_top_3_manufacturers = px.pie(top_3_manufacturers, values=top_3_manufacturers.values, names=top_3_manufacturers.index, title='Top 3 Handset Manufacturers')

    return html.Div([
        dcc.Graph(figure=fig_top_10_handsets),
        dcc.Graph(figure=fig_top_3_manufacturers)
    ])

# User Engagement Analysis
def get_user_engagement():
    # Aggregate engagement information
    df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    user_activity = df.groupby('MSISDN/Number').agg(
        total_duration=('Dur. (ms)', 'sum'),
        total_data=('total_data', 'sum'),
        number_of_sessions=('Bearer Id', 'count')
    ).reset_index()

    # Calculate a combined activity score
    user_activity['activity_score'] = user_activity['total_duration'] + user_activity['total_data']

    # Standardize the activity score
    scaler = StandardScaler()
    user_activity['activity_score_scaled'] = scaler.fit_transform(user_activity[['activity_score']])

    # Run K-Means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_activity['engagement_cluster'] = kmeans.fit_predict(user_activity[['activity_score_scaled']])

    fig_engagement_clusters = px.histogram(user_activity, x='engagement_cluster', title='Distribution of Engagement Clusters', labels={'x': 'Engagement Cluster', 'y': 'Number of Users'})

    return html.Div([
        dcc.Graph(figure=fig_engagement_clusters)
    ])

# Experience Analysis
def get_experience_analysis():
    # Aggregate experience information
    experience_aggregates = df.groupby('MSISDN/Number').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        avg_rtt=('Avg RTT DL (ms)', 'mean'),  # Assuming you want the downlink RTT
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')  # Assuming you want the downlink throughput
    ).reset_index()

    # Prepare data for clustering
    X_experience = experience_aggregates[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
    scaler = StandardScaler()
    X_experience_scaled = scaler.fit_transform(X_experience)

    # Run K-Means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    experience_aggregates['experience_cluster'] = kmeans.fit_predict(X_experience_scaled)

    fig_experience_clusters = px.scatter(experience_aggregates, x='avg_tcp_retransmission', y='avg_rtt', color='experience_cluster', title='Cluster Visualization: Average TCP Retransmission vs. Average RTT', labels={'x': 'Average TCP Retransmission (Bytes)', 'y': 'Average RTT (ms)'})

    return html.Div([
        dcc.Graph(figure=fig_experience_clusters)
    ])

# Satisfaction Analysis
def get_satisfaction_analysis():
    # Merge engagement and experience data
    ##combined_data = pd.merge(engagement_aggregates, experience_aggregates, on='MSISDN/Number')

    # Calculate satisfaction score as the average of engagement and experience scores
    combined_data['satisfaction_score'] = (combined_data['engagement_score'] + combined_data['experience_score']) / 2

    # Identify the top 10 satisfied customers
    top_satisfied_customers = combined_data.nlargest(10, 'satisfaction_score')

    # Run K-Means clustering with k=2 on engagement and experience scores
    kmeans_satisfaction = KMeans(n_clusters=2, random_state=42)
    combined_data['satisfaction_cluster'] = kmeans_satisfaction.fit_predict(combined_data[['engagement_score', 'experience_score']])

    # Aggregate average satisfaction and experience scores per cluster
    cluster_aggregates = combined_data.groupby('satisfaction_cluster').agg(
        avg_satisfaction_score=('satisfaction_score', 'mean'),
        avg_experience_score=('experience_score', 'mean')
    ).reset_index()

    fig_top_satisfied_customers = px.bar(top_satisfied_customers, x='MSISDN/Number', y='satisfaction_score', title='Top 10 Satisfied Customers', labels={'x': 'Customer ID', 'y': 'Satisfaction Score'})
    fig_satisfaction_clusters = px.scatter(combined_data, x='engagement_score', y='experience_score', color='satisfaction_cluster', title='Satisfaction Clusters Visualization', labels={'x': 'Engagement Score', 'y': 'Experience Score'})
    fig_cluster_aggregates = px.bar(cluster_aggregates, x='satisfaction_cluster', y=['avg_satisfaction_score', 'avg_experience_score'], title='Average Satisfaction and Experience Scores per Cluster', labels={'satisfaction_cluster': 'Satisfaction Cluster'})

    return html.Div([
        dcc.Graph(figure=fig_top_satisfied_customers),
        dcc.Graph(figure=fig_satisfaction_clusters),
        dcc.Graph(figure=fig_cluster_aggregates)
    ])

# Create Dash app
app = dash.Dash(__name__)

# Layout of the dashboard with tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='User Overview Analysis', children=[
            get_user_overview()
        ]),
        dcc.Tab(label='User Engagement Analysis', children=[
            get_user_engagement()
        ]),
        dcc.Tab(label='Experience Analysis', children=[
            get_experience_analysis()
        ]),
        ##dcc.Tab(label='Satisfaction Analysis', children=[
        ##    get_satisfaction_analysis()
        ##])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)