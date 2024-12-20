import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

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

# Initial inspection
print(df.info())
print(df.describe())
print(df.head())

# Identify columns with missing values
missing_cols = df.columns[df.isnull().any()]
print("Columns with missing values:\n", missing_cols)

# Fill missing values with mean (only for numeric columns)
for col in missing_cols:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)  # For categorical data, fill with mode

# Verify no more missing values
print("Missing values after filling:\n", df.isnull().sum())
