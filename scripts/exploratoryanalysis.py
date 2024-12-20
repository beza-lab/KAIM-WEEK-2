import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Identify and replace outliers using IQR (for numeric columns only)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Replace outliers with mean
for col in numeric_cols:
    df.loc[(df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3 + 1.5 * IQR[col])), col] = df[col].mean()

# Describe all relevant variables and their data types
variable_descriptions = df.dtypes
print(variable_descriptions)

# Basic Metrics Analysis
basic_metrics = df.describe()
print("Basic Metrics:\n", basic_metrics)

# Segment the users into decile classes based on total session duration
df['decile_class'], bins = pd.qcut(df['Dur. (ms)'].rank(method='first'), 10, labels=False, retbins=True)
print("Bins for decile classes:\n", bins)

# Compute total data (DL+UL) per decile class
df['total_dl_ul'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
decile_data = df.groupby('decile_class').agg(
    total_dl_ul=('total_dl_ul', 'sum')
).reset_index()
print(decile_data)

# Non-Graphical Univariate Analysis
dispersion_parameters = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print("Dispersion Parameters:\n", dispersion_parameters)

# Graphical Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Total DL (Bytes)'], kde=True)
plt.title('Distribution of Total DL (Bytes)')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Youtube DL (Bytes)', y='total_dl_ul', data=df)
plt.title('Relationship between YouTube DL and Total Data')
plt.show()

# Correlation Analysis
correlation_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                         'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Dimensionality Reduction (PCA)
features = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
x = df[features]
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print(pca_df.head())

# Explained variance
print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)

# Visualization
# Top 10 Handsets
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette="viridis")
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
    sns.barplot(x=handsets.values, y=handsets.index, palette="viridis")
    plt.title(f'Top 5 Handsets for {manufacturer}')
    plt.xlabel('Number of Users')
    plt.ylabel('Handset')
    plt.show()
