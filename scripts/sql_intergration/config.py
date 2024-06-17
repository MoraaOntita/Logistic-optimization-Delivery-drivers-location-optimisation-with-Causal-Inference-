# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL configuration
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Define CSV file paths and table names
CSV_FILE_1 = '/home/moraa/Documents/10_academy/Week-8/data/driver_locations_during_request.csv'
CSV_FILE_2 = '/home/moraa/Documents/10_academy/Week-8/data/nb.csv'
TABLE_1 = 'df1_driver_locations_during_request'
TABLE_2 = 'df2_nb'
