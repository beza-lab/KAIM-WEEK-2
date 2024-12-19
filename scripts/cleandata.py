import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Load data
df = pd.read_csv('D:/week2 data/Copy of Week2_challenge_data_source(CSV).csv')

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Percentage of Missing Values:\n", missing_percentage)

# Define columns to drop (complete list, no ellipsis)
columns_to_drop = [
    'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 
    'TCP UL Retrans. Vol (Bytes)', 'Nb of sec with 125000B < Vol DL', 
    'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 31250B < Vol DL < 125000B', 
    'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B', 
    'Nb of sec with 6250B < Vol UL < 37500B'
]

# Drop columns
df_cleaned = df.drop(columns=columns_to_drop)

# Mean/Median/Mode Imputation for remaining columns
df_cleaned.loc[:, 'Start ms'] = df_cleaned['Start ms'].fillna(df_cleaned['Start ms'].mean())
df_cleaned.loc[:, 'End ms'] = df_cleaned['End ms'].fillna(df_cleaned['End ms'].median())
df_cleaned.loc[:, 'Handset Manufacturer'] = df_cleaned['Handset Manufacturer'].fillna(df_cleaned['Handset Manufacturer'].mode()[0])

# Forward and Backward Fill
df_cleaned = df_cleaned.ffill()
df_cleaned = df_cleaned.bfill()

# Interpolation for time series data
df_cleaned = df_cleaned.interpolate(method='linear')

# Handle zero value columns by replacing zeros with NaN, then applying imputation
zero_columns = [
    'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)'
]
df_cleaned[zero_columns] = df_cleaned[zero_columns].replace(0, np.nan)

# Apply mean imputation to these columns (you can adjust the strategy as needed)
for column in zero_columns:
    df_cleaned.loc[:, column] = df_cleaned[column].fillna(df_cleaned[column].mean())

# Print cleaned data summary
print("Cleaned Data Summary:\n", df_cleaned.isnull().sum())

# Display the first few rows of the cleaned DataFrame to verify data
print(df_cleaned.head())

# Check the data types of the DataFrame
print(df_cleaned.dtypes)

# Identify Outliers using IQR (Ensure columns are numeric)
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
Q1 = df_cleaned[numeric_cols].quantile(0.25)
Q3 = df_cleaned[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR)))

# Display the number of outliers per column
print("Number of outliers in each column:\n", outliers.sum())

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('clean_data.csv', index=False)

# Visualize Outliers using Boxplots
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_cleaned[numeric_cols], orient='h')
plt.title('Boxplot for Outlier Detection')
plt.show()