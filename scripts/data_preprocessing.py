import pandas as pd
import logging
from config.config import Config
from typing import Tuple

def setup_logging() -> None:
    """Sets up the logging configuration."""
    logging.basicConfig(
        filename=Config.LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the data from the specified file paths.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The loaded dataframes.
    """
    try:
        df1 = pd.read_csv(Config.DF1_PATH)
        df2 = pd.read_csv(Config.DF2_PATH)
        logging.info("Data loaded successfully.")
        return df1, df2
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

def handle_missing_values(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handles missing values in the dataframes.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The dataframes with missing values handled.
    """
    try:
        # Drop specified columns in df1
        df1.drop(columns=Config.DF1_DROP_COLUMNS, inplace=True)
        logging.info("Dropped specified columns from df1.")

        # Impute missing values in df2
        for column, method in Config.DF2_IMPUTE_COLUMNS.items():
            if method == 'mode':
                df2[column] = df2[column].fillna(df2[column].mode()[0])
        logging.info("Imputed missing values in df2.")

        return df1, df2
    except KeyError as e:
        logging.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while handling missing values: {e}")
        raise

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data by loading and handling missing values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The preprocessed dataframes.
    """
    df1, df2 = load_data()
    df1, df2 = handle_missing_values(df1, df2)
    return df1, df2
