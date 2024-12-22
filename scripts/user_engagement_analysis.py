import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

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

# Calculate total traffic per application per user
df['social_media_traffic'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
df['google_traffic'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
df['email_traffic'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)']
df['youtube_traffic'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
df['netflix_traffic'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
df['gaming_traffic'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
df['other_traffic'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']

# Aggregate total traffic per application per user
user_traffic = df.groupby('MSISDN/Number').agg(
    total_social_media_traffic=('social_media_traffic', 'sum'),
    total_google_traffic=('google_traffic', 'sum'),
    total_email_traffic=('email_traffic', 'sum'),
    total_youtube_traffic=('youtube_traffic', 'sum'),
    total_netflix_traffic=('netflix_traffic', 'sum'),
    total_gaming_traffic=('gaming_traffic', 'sum'),
    total_other_traffic=('other_traffic', 'sum')
).reset_index()

# Determine the top 3 most used applications
total_traffic_per_application = user_traffic[['total_social_media_traffic', 'total_google_traffic', 'total_email_traffic', 'total_youtube_traffic', 'total_netflix_traffic', 'total_gaming_traffic', 'total_other_traffic']].sum()
top_3_applications = total_traffic_per_application.nlargest(3)
print("Top 3 Most Used Applications:\n", top_3_applications)

# Plot the top 3 most used applications
plt.figure(figsize=(10, 6))
sns.barplot(x=top_3_applications.values, y=top_3_applications.index, hue=top_3_applications.values, palette="viridis", dodge=False, legend=False)
plt.title('Top 3 Most Used Applications')
plt.xlabel('Total Traffic (Bytes)')
plt.ylabel('Application')
plt.show()

# Identify top 10 most engaged users per application
top_10_social_media_users = user_traffic.nlargest(10, 'total_social_media_traffic')
top_10_google_users = user_traffic.nlargest(10, 'total_google_traffic')
top_10_email_users = user_traffic.nlargest(10, 'total_email_traffic')
top_10_youtube_users = user_traffic.nlargest(10, 'total_youtube_traffic')
top_10_netflix_users = user_traffic.nlargest(10, 'total_netflix_traffic')
top_10_gaming_users = user_traffic.nlargest(10, 'total_gaming_traffic')
top_10_other_users = user_traffic.nlargest(10, 'total_other_traffic')

print("Top 10 Social Media Users:\n", top_10_social_media_users)
print("Top 10 Google Users:\n", top_10_google_users)
print("Top 10 Email Users:\n", top_10_email_users)
print("Top 10 YouTube Users:\n", top_10_youtube_users)
print("Top 10 Netflix Users:\n", top_10_netflix_users)
print("Top 10 Gaming Users:\n", top_10_gaming_users)
print("Top 10 Other Users:\n", top_10_other_users)

# Visualization: Top 10 Most Engaged Users per Application
def plot_top_users(data, title, xlabel):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data.iloc[:, 1].values, y=data['MSISDN/Number'], hue=data.iloc[:, 1].values, palette="viridis", dodge=False, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('User ID')
    plt.show()

# Plot for top applications
plot_top_users(top_10_gaming_users, 'Top 10 Most Engaged Users in Gaming', 'Total Gaming Traffic (Bytes)')
plot_top_users(top_10_other_users, 'Top 10 Most Engaged Users in Other Applications', 'Total Other Traffic (Bytes)')
plot_top_users(top_10_youtube_users, 'Top 10 Most Engaged Users in YouTube', 'Total YouTube Traffic (Bytes)')


# Calculate user activity based on total duration and total data usage
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

# Analyze Clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
clusters = pd.DataFrame(cluster_centers, columns=['activity_score'])
print("Cluster Centers:\n", clusters)

# Plot the distribution of engagement clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='engagement_cluster', hue='engagement_cluster', data=user_activity, palette="viridis", legend=False)
plt.title('Distribution of Engagement Clusters')
plt.xlabel('Engagement Cluster')
plt.ylabel('Number of Users')
plt.show()