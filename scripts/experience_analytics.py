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

# Handle missing values (replace with mean for numeric and mode for categorical)
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype == 'float64' else col.fillna(col.mode()[0]))

# Handle outliers (replace with mean)
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), col] = df[col].mean()

# Aggregate information per customer
customer_aggregates = df.groupby('MSISDN/Number').agg(
    avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
    avg_rtt=('Avg RTT DL (ms)', 'mean'),  # Assuming you want the downlink RTT
    handset_type=('Handset Type', lambda x: x.mode()[0]),
    avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')  # Assuming you want the downlink throughput
).reset_index()

print(customer_aggregates.head())

# Top, bottom, and most frequent values
def compute_top_bottom_frequent(df, column):
    top_values = df[column].nlargest(10)
    bottom_values = df[column].nsmallest(10)
    most_frequent_values = df[column].value_counts().head(10)
    return top_values, bottom_values, most_frequent_values

# TCP values
top_tcp, bottom_tcp, frequent_tcp = compute_top_bottom_frequent(df, 'TCP DL Retrans. Vol (Bytes)')

# RTT values
top_rtt, bottom_rtt, frequent_rtt = compute_top_bottom_frequent(df, 'Avg RTT DL (ms)')

# Throughput values
top_throughput, bottom_throughput, frequent_throughput = compute_top_bottom_frequent(df, 'Avg Bearer TP DL (kbps)')

print("Top TCP values:\n", top_tcp)
print("Bottom TCP values:\n", bottom_tcp)
print("Most Frequent TCP values:\n", frequent_tcp)

print("Top RTT values:\n", top_rtt)
print("Bottom RTT values:\n", bottom_rtt)
print("Most Frequent RTT values:\n", frequent_rtt)

print("Top Throughput values:\n", top_throughput)
print("Bottom Throughput values:\n", bottom_throughput)
print("Most Frequent Throughput values:\n", frequent_throughput)

# Distribution of Average Throughput per Handset Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='handset_type', y='avg_throughput', data=customer_aggregates)
plt.xticks(rotation=90)
plt.title('Distribution of Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput (kbps)')
plt.show()

# Distribution of Average TCP Retransmission per Handset Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='handset_type', y='avg_tcp_retransmission', data=customer_aggregates)
plt.xticks(rotation=90)
plt.title('Distribution of Average TCP Retransmission per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission (Bytes)')
plt.show()

# Prepare data for clustering
X = customer_aggregates[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
customer_aggregates['experience_cluster'] = kmeans.fit_predict(X_scaled)

# Analyze Clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
clusters = pd.DataFrame(cluster_centers, columns=['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput'])
print("Cluster Centers:\n", clusters)

# Description of Each Cluster
for i in range(3):
    print(f"Cluster {i}:")
    cluster_data = customer_aggregates[customer_aggregates['experience_cluster'] == i]
    print(cluster_data.describe())
    print("\n")

# Visualization of Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_tcp_retransmission', y='avg_rtt', hue='experience_cluster', data=customer_aggregates, palette="viridis")
plt.legend(loc='upper right') 
plt.title('Cluster Visualization: Average TCP Retransmission vs. Average RTT')
plt.xlabel('Average TCP Retransmission (Bytes)')
plt.ylabel('Average RTT (ms)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_tcp_retransmission', y='avg_throughput', hue='experience_cluster', data=customer_aggregates, palette="viridis")
plt.legend(loc='upper right') 
plt.title('Cluster Visualization: Average TCP Retransmission vs. Average Throughput')
plt.xlabel('Average TCP Retransmission (Bytes)')
plt.ylabel('Average Throughput (kbps)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_rtt', y='avg_throughput', hue='experience_cluster', data=customer_aggregates, palette="viridis")
plt.legend(loc='upper right') 
plt.title('Cluster Visualization: Average RTT vs. Average Throughput')
plt.xlabel('Average RTT (ms)')
plt.ylabel('Average Throughput (kbps)')
plt.show()