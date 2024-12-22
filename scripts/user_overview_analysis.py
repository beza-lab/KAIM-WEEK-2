import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Basic Metrics
print(df.describe())

# Identify the top 10 handsets used by the customers
top_10_handsets = df['Handset Type'].value_counts().head(10)
print("Top 10 Handsets:\n", top_10_handsets)

# Identify the top 3 handset manufacturers
top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
print("Top 3 Handset Manufacturers:\n", top_3_manufacturers)

# Identify the top 5 handsets per top 3 manufacturers
top_5_handsets_per_manufacturer = {}

for manufacturer in top_3_manufacturers.index:
    top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
    top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

print("Top 5 Handsets per Top 3 Manufacturers:\n", top_5_handsets_per_manufacturer)

# Visualization
# Top 10 Handsets
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, hue=top_10_handsets.index, palette="viridis", dodge=False, legend=False)
plt.title('Top 10 Handsets Used by Customers')
plt.xlabel('Number of Users')
plt.ylabel('Handset')
plt.show()

# Top 3 Handset Manufacturers
plt.figure(figsize=(8, 8))
top_3_manufacturers.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", 3))
plt.title('Top 3 Handset Manufacturers')
plt.ylabel('')
plt.show()

# Top 5 Handsets per Top 3 Manufacturers
for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
    plt.figure(figsize=(10, 6))
    sns.barplot(x=handsets.values, y=handsets.index, hue=handsets.index, palette="viridis", dodge=False, legend=False)
    plt.title(f'Top 5 Handsets for {manufacturer}')
    plt.xlabel('Number of Users')
    plt.ylabel('Handset')
    plt.show()

# Data Usage Analysis: Bivariate Analysis
df['youtube_dl'] = pd.to_numeric(df['Youtube DL (Bytes)'], errors='coerce').fillna(0)
df['total_dl_ul'] = pd.to_numeric(df['Total DL (Bytes)'] + df['Total UL (Bytes)'], errors='coerce').fillna(0)

# Scatter Plot: YouTube Download vs Total Data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='youtube_dl', y='total_dl_ul', data=df)
plt.title('Relationship between YouTube DL and Total Data')
plt.xlabel('YouTube Download (Bytes)')
plt.ylabel('Total Data (Bytes)')
plt.show()