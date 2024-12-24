import pandas as pd
import numpy as np
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

# Prepare data for User Overview Analysis
top_10_handsets = df['Handset Type'].value_counts().head(10)
top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

top_5_handsets_per_manufacturer = {}
for manufacturer in top_3_manufacturers.index:
    top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
    top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

# Prepare data for User Engagement Analysis
df['youtube_dl'] = pd.to_numeric(df['Youtube DL (Bytes)'], errors='coerce').fillna(0)
df['total_dl_ul'] = pd.to_numeric(df['Total DL (Bytes)'] + df['Total UL (Bytes)'], errors='coerce').fillna(0)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Data Insights Dashboard"

# Define the layout of the app
app.layout = html.Div(children=[
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='User Overview Analysis', value='tab-1'),
        dcc.Tab(label='User Engagement Analysis', value='tab-2'),
        dcc.Tab(label='Experience Analysis', value='tab-3'),
        dcc.Tab(label='Satisfaction Analysis', value='tab-4'),
    ]),
    html.Div(id='tabs-content')
])

# Define callback to update the content of each tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('User Overview Analysis'),
            dcc.Graph(
                figure=px.bar(x=top_10_handsets.values, y=top_10_handsets.index, title='Top 10 Handsets Used by Customers', labels={'x': 'Number of Users', 'y': 'Handset'})
            ),
            dcc.Graph(
                figure=px.pie(values=top_3_manufacturers.values, names=top_3_manufacturers.index, title='Top 3 Handset Manufacturers')
            ),
            html.H4('Top 5 Handsets per Top 3 Manufacturers'),
            *[html.Div(children=[
                html.H5(f'Top 5 Handsets for {manufacturer}'),
                dcc.Graph(
                    figure=px.bar(x=handsets.values, y=handsets.index, title=f'Top 5 Handsets for {manufacturer}', labels={'x': 'Number of Users', 'y': 'Handset'})
                )
            ]) for manufacturer, handsets in top_5_handsets_per_manufacturer.items()]
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('User Engagement Analysis'),
            dcc.Graph(
                figure=px.scatter(df, x='youtube_dl', y='total_dl_ul', title='Relationship between YouTube DL and Total Data', labels={'x': 'YouTube Download (Bytes)', 'y': 'Total Data (Bytes)'})
            )
        ])
    elif tab == 'tab-3':
        # Prepare data for Experience Analysis
        experience_aggregates = df.groupby('MSISDN/Number').agg(
            avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
            avg_rtt=('Avg RTT DL (ms)', 'mean'),
            avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
        ).reset_index()

        # Prepare data for clustering
        X_experience = experience_aggregates[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
        scaler = StandardScaler()
        X_experience_scaled = scaler.fit_transform(X_experience)
        kmeans_experience = KMeans(n_clusters=3, random_state=42)
        experience_aggregates['experience_cluster'] = kmeans_experience.fit_predict(X_experience_scaled)

        return html.Div([
            html.H3('Experience Analysis'),
            dcc.Graph(
                figure=px.scatter(experience_aggregates, x='avg_tcp_retransmission', y='avg_rtt', color='experience_cluster',
                                  title='Cluster Visualization: Average TCP Retransmission vs. Average RTT', labels={'x': 'Average TCP Retransmission (Bytes)', 'y': 'Average RTT (ms)'})
            )
        ])
    elif tab == 'tab-4':
        # Prepare data for Satisfaction Analysis
        engagement_aggregates = df.groupby('MSISDN/Number').agg(
            total_duration=('Dur. (ms)', 'sum'),
            total_data=('total_data', 'sum'),
            number_of_sessions=('Bearer Id', 'count')
        ).reset_index()
        engagement_aggregates['engagement_score'] = engagement_aggregates['total_duration'] + engagement_aggregates['total_data']

        experience_aggregates = df.groupby('MSISDN/Number').agg(
            avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
            avg_rtt=('Avg RTT DL (ms)', 'mean'),
            avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
        ).reset_index()
        X_experience = experience_aggregates[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
        scaler = StandardScaler()
        X_experience_scaled = scaler.fit_transform(X_experience)
        kmeans_experience = KMeans(n_clusters=3, random_state=42)
        experience_aggregates['experience_cluster'] = kmeans_experience.fit_predict(X_experience_scaled)

        engagement_aggregates['engagement_score'] = engagement_aggregates.apply(
            lambda row: np.sqrt(np.sum((row[['total_duration', 'total_data', 'number_of_sessions']] - kmeans_experience.cluster_centers_[np.argmin(kmeans_experience.cluster_centers_[:, 2])]) ** 2)), axis=1)
        experience_aggregates['experience_score'] = experience_aggregates.apply(
            lambda row: np.sqrt(np.sum((row[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']] - kmeans_experience.cluster_centers_[np.argmax(kmeans_experience.cluster_centers_[:, 0])]) ** 2)), axis=1)

        combined_data = pd.merge(engagement_aggregates, experience_aggregates, on='MSISDN/Number')
        combined_data['satisfaction_score'] = (combined_data['engagement_score'] + combined_data['experience_score']) / 2
        top_satisfied_customers = combined_data.nlargest(10, 'satisfaction_score')
        kmeans_satisfaction = KMeans(n_clusters=2, random_state=42)
        combined_data['satisfaction_cluster'] = kmeans_satisfaction.fit_predict(combined_data[['engagement_score', 'experience_score']])

        return html.Div([
            html.H3('Satisfaction Analysis'),
            dcc.Graph(
                figure=px.bar(top_satisfied_customers, x='MSISDN/Number', y='satisfaction_score',
                              title='Top 10 Satisfied Customers', labels={'x': 'Customer ID', 'y': 'Satisfaction Score'})
            ),
            dcc.Graph(
                figure=px.scatter(combined_data, x='engagement_score', y='experience_score', color='satisfaction_cluster',
                                  title='Satisfaction Clusters Visualization', labels={'x': 'Engagement Score', 'y': 'Experience Score'})
            )
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
