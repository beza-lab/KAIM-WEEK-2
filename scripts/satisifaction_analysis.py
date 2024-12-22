import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Handle missing values (replace with mean for numeric and mode for categorical)
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype == 'float64' else col.fillna(col.mode()[0]))

# Handle outliers (replace with mean)
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), col] = df[col].mean()

# Create a column for total data usage
df['total_data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

# Aggregate engagement information per customer
engagement_aggregates = df.groupby('MSISDN/Number').agg(
    total_duration=('Dur. (ms)', 'sum'),
    total_data=('total_data', 'sum'),
    number_of_sessions=('Bearer Id', 'count')
).reset_index()

# Calculate a combined engagement score
engagement_aggregates['engagement_score'] = engagement_aggregates['total_duration'] + engagement_aggregates['total_data']

# Aggregate experience information per customer
experience_aggregates = df.groupby('MSISDN/Number').agg(
    avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
    avg_rtt=('Avg RTT DL (ms)', 'mean'),  # Assuming you want the downlink RTT
    avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')  # Assuming you want the downlink throughput
).reset_index()

# Prepare data for clustering
X_experience = experience_aggregates[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
X_engagement = engagement_aggregates[['total_duration', 'total_data', 'number_of_sessions']]

# Standardize the data
scaler = StandardScaler()
X_experience_scaled = scaler.fit_transform(X_experience)
X_engagement_scaled = scaler.fit_transform(X_engagement)

# Run K-Means clustering with k=3 for experience
kmeans_experience = KMeans(n_clusters=3, random_state=42)
experience_aggregates['experience_cluster'] = kmeans_experience.fit_predict(X_experience_scaled)

# Run K-Means clustering with k=3 for engagement
kmeans_engagement = KMeans(n_clusters=3, random_state=42)
engagement_aggregates['engagement_cluster'] = kmeans_engagement.fit_predict(X_engagement_scaled)

# Determine the less engaged cluster (cluster with the lowest average engagement score)
less_engaged_cluster_center = kmeans_engagement.cluster_centers_[np.argmin(kmeans_engagement.cluster_centers_[:, 2])]
# Determine the worst experience cluster (cluster with the highest average TCP retransmission)
worst_experience_cluster_center = kmeans_experience.cluster_centers_[np.argmax(kmeans_experience.cluster_centers_[:, 0])]

# Calculate engagement and experience scores based on Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

engagement_aggregates['engagement_score'] = engagement_aggregates.apply(
    lambda row: euclidean_distance(row[['total_duration', 'total_data', 'number_of_sessions']], less_engaged_cluster_center), axis=1)

experience_aggregates['experience_score'] = experience_aggregates.apply(
    lambda row: euclidean_distance(row[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']], worst_experience_cluster_center), axis=1)

# Merge engagement and experience data
combined_data = pd.merge(engagement_aggregates, experience_aggregates, on='MSISDN/Number')

# Calculate satisfaction score as the average of engagement and experience scores
combined_data['satisfaction_score'] = (combined_data['engagement_score'] + combined_data['experience_score']) / 2

# Task 4.2 - Identify the top 10 satisfied customers
top_satisfied_customers = combined_data.nlargest(10, 'satisfaction_score')
print("Top 10 Satisfied Customers:\n", top_satisfied_customers)

# Task 4.3 - Build a regression model to predict satisfaction score
X = combined_data[['engagement_score', 'experience_score']]
y = combined_data['satisfaction_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Regression Model Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Task 4.4 - Run K-Means clustering with k=2 on engagement and experience scores
kmeans_satisfaction = KMeans(n_clusters=2, random_state=42)
combined_data['satisfaction_cluster'] = kmeans_satisfaction.fit_predict(combined_data[['engagement_score', 'experience_score']])

# Task 4.5 - Aggregate average satisfaction and experience scores per cluster
cluster_aggregates = combined_data.groupby('satisfaction_cluster').agg(
    avg_satisfaction_score=('satisfaction_score', 'mean'),
    avg_experience_score=('experience_score', 'mean')
).reset_index()

print("Cluster Aggregates:\n", cluster_aggregates)

# Visualization of Satisfaction Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='engagement_score', y='experience_score', hue='satisfaction_cluster', data=combined_data, palette="viridis")
plt.legend(loc='upper right')  # Specify fixed legend location
plt.title('Satisfaction Clusters Visualization')
plt.xlabel('Engagement Score')
plt.ylabel('Experience Score')
plt.show()