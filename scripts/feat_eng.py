import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple
from config.config import Config

# Configure logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Trip Start Time is in datetime format."""
    try:
        df['Trip Start Time'] = pd.to_datetime(df['Trip Start Time'])
        logging.info("Preprocessed Trip Start Time to datetime format.")
    except Exception as e:
        logging.error(f"Error preprocessing Trip Start Time: {e}")
        raise
    return df

def extract_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day of the week from Trip Start Time."""
    try:
        df['Day of Week'] = df['Trip Start Time'].dt.day_name()
        logging.info("Extracted Day of Week from Trip Start Time.")
    except Exception as e:
        logging.error(f"Error extracting Day of Week: {e}")
        raise
    return df

def categorize_time_of_day(hour: int) -> str:
    """Categorize time of day."""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def extract_hour_and_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour from Trip Start Time and create Time of Day feature."""
    try:
        df['Hour'] = df['Trip Start Time'].dt.hour
        df['Time of Day'] = df['Hour'].apply(categorize_time_of_day)
        logging.info("Extracted Hour and Time of Day from Trip Start Time.")
    except Exception as e:
        logging.error(f"Error extracting Hour and Time of Day: {e}")
        raise
    return df

def create_is_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature indicating if the date is a holiday."""
    try:
        all_holidays = pd.to_datetime(Config.HOLIDAYS_2021 + Config.HOLIDAYS_2022)
        df['Is Holiday'] = df['Trip Start Time'].dt.date.isin(all_holidays.date)
        logging.info("Created Is Holiday feature.")
    except Exception as e:
        logging.error(f"Error creating Is Holiday feature: {e}")
        raise
    return df

def preprocess_trip_times(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Trip Start Time and Trip End Time are in datetime format."""
    try:
        df['Trip Start Time'] = pd.to_datetime(df['Trip Start Time'])
        df['Trip End Time'] = pd.to_datetime(df['Trip End Time'])
        logging.info("Preprocessed Trip Start Time and Trip End Time to datetime format.")
    except Exception as e:
        logging.error(f"Error preprocessing Trip Start Time and Trip End Time: {e}")
        raise
    return df

def split_origin_destination(df: pd.DataFrame) -> pd.DataFrame:
    """Split Trip Origin and Trip Destination into latitude and longitude."""
    try:
        df[['Origin Lat', 'Origin Lng']] = df['Trip Origin'].str.split(',', expand=True).astype(float)
        df[['Destination Lat', 'Destination Lng']] = df['Trip Destination'].str.split(',', expand=True).astype(float)
        logging.info("Split Trip Origin and Trip Destination into latitude and longitude.")
    except Exception as e:
        logging.error(f"Error splitting Trip Origin and Trip Destination: {e}")
        raise
    return df

def extract_additional_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional time-based features."""
    try:
        df['Start Hour'] = df['Trip Start Time'].dt.hour
        df['Start Day of Week'] = df['Trip Start Time'].dt.dayofweek
        logging.info("Extracted additional time-based features.")
    except Exception as e:
        logging.error(f"Error extracting additional time-based features: {e}")
        raise
    return df

def calculate_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trip duration in minutes."""
    try:
        df['Trip Duration'] = (df['Trip End Time'] - df['Trip Start Time']).dt.total_seconds() / 60
        logging.info("Calculated trip duration in minutes.")
    except Exception as e:
        logging.error(f"Error calculating trip duration: {e}")
        raise
    return df

def merge_and_calculate_distances(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge datasets and calculate distances."""
    try:
        df1.rename(columns={'order_id': 'Trip ID'}, inplace=True)
        df_merged = pd.merge(df1, df2, on='Trip ID')

        def calculate_distance(row, lat1, lon1, lat2, lon2):
            point1 = (row[lat1], row[lon1])
            point2 = (row[lat2], row[lon2])
            return geodesic(point1, point2).kilometers

        df_merged['Driver Distance to Origin'] = df_merged.apply(lambda row: calculate_distance(row, 'lat', 'lng', 'Origin Lat', 'Origin Lng'), axis=1)
        df_merged['Trip Distance'] = df_merged.apply(lambda row: calculate_distance(row, 'Origin Lat', 'Origin Lng', 'Destination Lat', 'Destination Lng'), axis=1)
        logging.info("Merged datasets and calculated distances.")
    except Exception as e:
        logging.error(f"Error merging datasets and calculating distances: {e}")
        raise
    return df_merged

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant features for scaling and apply StandardScaler."""
    try:
        features_to_scale = ['lat', 'lng', 'Origin Lat', 'Origin Lng', 'Destination Lat', 'Destination Lng', 'Trip Duration', 'Driver Distance to Origin', 'Trip Distance']
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        logging.info("Scaled selected features using StandardScaler.")
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        raise
    return df
