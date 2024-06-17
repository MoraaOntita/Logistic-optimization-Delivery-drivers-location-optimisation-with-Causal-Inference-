import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables from .env file
load_dotenv()

# Import configuration from config.py
from scripts.sql_intergration.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, CSV_FILE_1, CSV_FILE_2, TABLE_1, TABLE_2


# PostgreSQL connection function decorator
def with_postgres_connection(func):
    def wrapper(*args, **kwargs):
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        try:
            result = func(conn, *args, **kwargs)
        finally:
            conn.close()
        return result
    return wrapper


# Function to load data from CSV file into PostgreSQL table
@with_postgres_connection
def load_csv_to_db(conn: psycopg2.Connection, csv_file: str, table_name: str) -> None:
    """
    Load data from a CSV file into a PostgreSQL table.

    Args:
    - conn: psycopg2 connection object
    - csv_file: path to the CSV file
    - table_name: name of the PostgreSQL table
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Prepare columns for SQL insertion
    columns = df.columns.tolist()
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))
    
    # Create the insert query dynamically
    insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
    
    # Iterate over the DataFrame rows and execute the insert query
    records = df.values.tolist()
    with conn.cursor() as cur:
        cur.executemany(insert_query, records)
    
    # Commit the transaction
    conn.commit()


# Main function to execute data loading process
def main() -> None:
    """
    Main function to execute the data loading process.
    """
    try:
        # Load data into table 1
        load_csv_to_db(CSV_FILE_1, TABLE_1)
        print(f"Data loaded from {CSV_FILE_1} into table {TABLE_1}")

        # Load data into table 2
        load_csv_to_db(CSV_FILE_2, TABLE_2)
        print(f"Data loaded from {CSV_FILE_2} into table {TABLE_2}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
