import logging
from data_preprocessing import preprocess_data, setup_logging

def main() -> None:
    """
    The main function that runs the data preprocessing.
    """
    setup_logging()
    try:
        df1, df2 = preprocess_data()
        print("DataFrame 1 after preprocessing:")
        print(df1.head())
        print("\nDataFrame 2 after preprocessing:")
        print(df2.head())
        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
