import logging
from data_preprocessing import preprocess_data, setup_logging
import pandas as pd
from feat_eng import *
from config.config import Config
from feat_eng import preprocess_data, perform_feature_engineering
from analysis import perform_analysis
import logging
from feat_eng import preprocess_datetime, extract_day_of_week, extract_hour_and_time_of_day, create_is_holiday_feature, preprocess_trip_times, split_origin_destination, extract_additional_time_features, calculate_trip_duration, merge_and_calculate_distances, scale_features

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
    
    

def main():
    """Main function to execute feature engineering steps."""
    try:
        # Load datasets
        df1 = pd.read_csv(Config.DF1_PATH)
        df2 = pd.read_csv(Config.DF2_PATH)
        logging.info("Loaded datasets.")

        # Perform feature engineering
        df2 = preprocess_datetime(df2)
        df2 = extract_day_of_week(df2)
        df2 = extract_hour_and_time_of_day(df2)
        df2 = create_is_holiday_feature(df2)
        df2 = preprocess_trip_times(df2)
        df2 = split_origin_destination(df2)
        df2 = extract_additional_time_features(df2)
        df2 = calculate_trip_duration(df2)
        df_merged = merge_and_calculate_distances(df1, df2)
        df_merged = scale_features(df_merged)

        # Display the first few rows of the transformed dataframe
        print(df_merged.head())
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Preprocess data and perform feature engineering
        df1 = pd.read_csv(Config.DF1_PATH)
        df2 = pd.read_csv(Config.DF2_PATH)
        
        df_feat_eng = preprocess_data(df1, df2)
        df_feat_eng = perform_feature_engineering(df_feat_eng)
        
        # Perform analysis
        df_analysis, riders_count = perform_analysis(df_feat_eng)
        
        # Save or further process df_analysis if needed
        df_analysis.to_csv("analysis_results.csv", index=False)
        
        logger.info("Analysis completed successfully.")
        logger.info(f"Number of riders within {Config.RADIUS} km of accepted orders: {riders_count}")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    # Example usage
    df_feat_eng = pd.read_csv(Config.FEATURE_ENG_DATA_PATH)
    df_analysis, riders_count = perform_analysis(df_feat_eng)
    # Further processing or saving
    print(df_analysis.head())
    print(f"Number of riders within {Config.RADIUS} km of accepted orders: {riders_count}")
    

def add_remaining_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add remaining features to the merged DataFrame."""
    try:
        # Add remaining feature engineering steps here
        # Example:
        # df['New Feature'] = ...
        pass
    except Exception as e:
        raise RuntimeError(f"Error adding remaining features: {e}")

def process_dataset(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Process the dataset using feature engineering functions."""
    try:
        df1 = preprocess_datetime(df1)
        df1 = extract_day_of_week(df1)
        df1 = extract_hour_and_time_of_day(df1)
        df1 = create_is_holiday_feature(df1)

        df2 = preprocess_trip_times(df2)
        df2 = split_origin_destination(df2)
        df2 = extract_additional_time_features(df2)
        df2 = calculate_trip_duration(df2)

        df_merged = merge_and_calculate_distances(df1, df2)
        df_merged = scale_features(df_merged)

        add_remaining_features(df_merged)

        return df_merged

    except Exception as e:
        raise RuntimeError(f"Error processing dataset: {e}")

def main():
    try:
        # Load datasets (replace with actual loading mechanism)
        df1 = pd.DataFrame({'order_id': [1, 2, 3], 'lat': [0, 1, 2], 'lng': [0, 1, 2]})
        df2 = pd.DataFrame({'Trip ID': [1, 2, 3], 'Trip Start Time': ['2023-01-01 08:00:00', '2023-01-02 12:00:00', '2023-01-03 18:00:00'], 'Trip End Time': ['2023-01-01 09:00:00', '2023-01-02 13:00:00', '2023-01-03 19:00:00'], 'Trip Origin': ['0.0,0.0', '1.0,1.0', '2.0,2.0'], 'Trip Destination': ['1.0,1.0', '2.0,2.0', '3.0,3.0']})

        # Process datasets
        df_processed = process_dataset(df1, df2)

        # Display processed DataFrame
        print(df_processed.head())

    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main()
