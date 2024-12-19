import pandas as pd

# Load cleaned data
df_cleaned = pd.read_csv('D:/week2 data/clean_data.csv')

# Aggregate per user the required information
user_aggregates = df_cleaned.groupby('MSISDN/Number').agg(
    number_of_xdr_sessions=('Bearer Id', 'count'),
    session_duration=('Dur. (ms)', 'sum'),
    total_dl_data=('Total DL (Bytes)', 'sum'),
    total_ul_data=('Total UL (Bytes)', 'sum'),
    social_media_dl=('Social Media DL (Bytes)', 'sum'),
    social_media_ul=('Social Media UL (Bytes)', 'sum'),
    google_dl=('Google DL (Bytes)', 'sum'),
    google_ul=('Google UL (Bytes)', 'sum'),
    email_dl=('Email DL (Bytes)', 'sum'),
    email_ul=('Email UL (Bytes)', 'sum'),
    youtube_dl=('Youtube DL (Bytes)', 'sum'),
    youtube_ul=('Youtube UL (Bytes)', 'sum'),
    netflix_dl=('Netflix DL (Bytes)', 'sum'),
    netflix_ul=('Netflix UL (Bytes)', 'sum'),
    gaming_dl=('Gaming DL (Bytes)', 'sum'),
    gaming_ul=('Gaming UL (Bytes)', 'sum'),
    other_dl=('Other DL (Bytes)', 'sum'),
    other_ul=('Other UL (Bytes)', 'sum')
).reset_index()

# Calculate total data volume for each application (DL + UL)
user_aggregates['total_social_media_data'] = user_aggregates['social_media_dl'] + user_aggregates['social_media_ul']
user_aggregates['total_google_data'] = user_aggregates['google_dl'] + user_aggregates['google_ul']
user_aggregates['total_email_data'] = user_aggregates['email_dl'] + user_aggregates['email_ul']
user_aggregates['total_youtube_data'] = user_aggregates['youtube_dl'] + user_aggregates['youtube_ul']
user_aggregates['total_netflix_data'] = user_aggregates['netflix_dl'] + user_aggregates['netflix_ul']
user_aggregates['total_gaming_data'] = user_aggregates['gaming_dl'] + user_aggregates['gaming_ul']
user_aggregates['total_other_data'] = user_aggregates['other_dl'] + user_aggregates['other_ul']

# Drop intermediate columns if desired
user_aggregates = user_aggregates.drop(columns=[
    'social_media_dl', 'social_media_ul', 'google_dl', 'google_ul', 
    'email_dl', 'email_ul', 'youtube_dl', 'youtube_ul', 
    'netflix_dl', 'netflix_ul', 'gaming_dl', 'gaming_ul', 
    'other_dl', 'other_ul'
])

# Display the first few rows of the aggregated DataFrame
print(user_aggregates.head())

# Save aggregated data to a new CSV file
output_aggregate_path = 'D:/week2 data/Aggregated_User_Data.csv'
user_aggregates.to_csv(output_aggregate_path, index=False)

print(f"Aggregated user data saved to {output_aggregate_path}")