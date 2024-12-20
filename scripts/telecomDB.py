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

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Identify and replace outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Replace outliers with mean
for col in df.select_dtypes(include=[np.number]).columns:
    df.loc[(df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3 + 1.5 * IQR[col])), col] = df[col].mean()

# Describe all relevant variables and their data types
variable_descriptions = df.dtypes
print(variable_descriptions)

# Ensure 'Dur. (ms)' is numeric and handle missing values
df['Dur. (ms)'] = pd.to_numeric(df['Dur. (ms)'], errors='coerce')
df['Dur. (ms)'].fillna(df['Dur. (ms)'].mean(), inplace=True)

# Verify the column is ready for binning
print("Summary of 'Dur. (ms)':\n", df['Dur. (ms)'].describe())

# Segment the users into decile classes based on total session duration
# Handle bin edges by ensuring unique values for binning
df['decile_class'], bins = pd.qcut(df['Dur. (ms)'].rank(method='first'), 10, labels=False, retbins=True)
print("Bins for decile classes:\n", bins)

# Compute total data (DL+UL) per decile class
df['total_dl_ul'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
decile_data = df.groupby('decile_class').agg(
    total_dl_ul=('total_dl_ul', 'sum')
).reset_index()
print(decile_data)

# Basic Metrics Analysis
basic_metrics = df.describe()
print("Basic Metrics:\n", basic_metrics)

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

# Save aggregated data
output_aggregate_path = 'D:/week2 data/Aggregated_User_Data.csv'
decile_data.to_csv(output_aggregate_path, index=False)
print(f"Aggregated user data saved to {output_aggregate_path}")

# Save cleaned data
output_cleaned_path = 'D:/week2 data/Cleaned_Week2_challenge_data_source.csv'
df.to_csv(output_cleaned_path, index=False)
print(f"Cleaned data saved to {output_cleaned_path}")

# Identify the top 10 handsets used by the customers
top_10_handsets = df['handset'].value_counts().head(10)
print("Top 10 Handsets:\n", top_10_handsets)

# Identify the top 3 handset manufacturers
top_3_manufacturers = df['handset_manufacturer'].value_counts().head(3)
print("Top 3 Handset Manufacturers:\n", top_3_manufacturers)

# Identify the top 5 handsets per top 3 manufacturers
top_5_handsets_per_manufacturer = {}
for manufacturer in top_3_manufacturers.index:
    top_5_handsets = df[df['handset_manufacturer'] == manufacturer]['handset'].value_counts().head(5)
    top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

print("Top 5 Handsets per Top 3 Manufacturers:\n", top_5_handsets_per_manufacturer)

# Interpretation and Recommendation
interpretation = """
Interpretation:
1. Top Handsets: The top 10 handsets indicate strong customer preferences towards specific models.
2. Top Manufacturers: The top 3 handset manufacturers dominate the market.
3. Top Handsets per Manufacturer: Understanding the most popular models within each leading manufacturer helps tailor marketing efforts.

Recommendations:
1. Targeted Marketing Campaigns: Focus marketing campaigns on the top handsets to maximize engagement and conversion.
2. Promotional Offers: Create promotional offers and discounts for the most popular models from the top manufacturers to drive sales.
3. Partnerships: Strengthen partnerships with the top manufacturers to leverage their brand loyalty and market presence.
4. Product Positioning: Use insights from the top handsets to guide product positioning and highlight features that resonate with customers.
"""
print(interpretation)
