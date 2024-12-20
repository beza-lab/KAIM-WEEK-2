import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

print(f"Database Name: {db_name}")
print(f"Database User: {db_user}")
print(f"Database Password: {db_password}")
print(f"Database Host: {db_host}")
print(f"Database Port: {db_port}")