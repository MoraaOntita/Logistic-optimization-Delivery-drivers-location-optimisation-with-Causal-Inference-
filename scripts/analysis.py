import pandas as pd
from geopy.distance import geodesic
from haversine import Unit, haversine
from config.config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_latitude(lat: float) -> float:
    """Clean latitude values."""
    if lat < -90:
        return -90
    elif lat > 90:
        return 90
    return lat

def compute_geodesic_distance(row: pd.Series) -> float:
    """Compute geodesic distance between origin and destination."""
    coords_1 = (row['Origin Lat'], row['Origin Lng'])
    coords_2 = (row['Destination Lat'], row['Destination Lng'])
    try:
        return geodesic(coords_1, coords_2).kilometers
    except Exception as e:
        logger.error(f"Error computing geodesic distance: {str(e)}")
        return None

def compute_haversine_distance(row: pd.Series) -> float:
    """Compute Haversine distance between origin and destination."""
    coords_1 = (row['Origin Lat'], row['Origin Lng'])
    coords_2 = (row['Destination Lat'], row['Destination Lng'])
    try:
        return haversine(coords_1, coords_2, unit=Unit.KILOMETERS)
    except Exception as e:
        logger.error(f"Error computing Haversine distance: {str(e)}")
        return None

def compute_driving_speed(row: pd.Series) -> float:
    """Compute driving speed based on distance and duration."""
    distance = row['Haversine Distance']
    duration = row['Trip Duration']
    try:
        return distance / (duration / 60) if duration > 0 else 0
    except ZeroDivisionError:
        logger.error("Error computing driving speed: division by zero")
        return None

def count_riders_within_radius(df_feat_eng: pd.DataFrame, radius: float = 0.5) -> int:
    """Count riders within a specified radius of accepted orders."""
    accepted_riders_in_circle = 0
    for _, order in df_feat_eng[df_feat_eng['driver_action'] == 'accepted'].iterrows():
        order_lat = order['lat']
        order_lon = order['lng']
        
        for _, rider in df_feat_eng.iterrows():
            rider_lat = rider['lat']
            rider_lon = rider['lng']
            try:
                distance = haversine((order_lat, order_lon), (rider_lat, rider_lon))
                if distance <= radius:
                    accepted_riders_in_circle += 1
            except Exception as e:
                logger.error(f"Error computing haversine distance for rider: {str(e)}")
    
    return accepted_riders_in_circle

def perform_analysis(df_feat_eng: pd.DataFrame) -> pd.DataFrame:
    """Perform analysis on the feature-engineered DataFrame."""
    try:
        # Clean latitude values
        df_feat_eng['Origin Lat'] = df_feat_eng['Origin Lat'].apply(clean_latitude)
        df_feat_eng['Destination Lat'] = df_feat_eng['Destination Lat'].apply(clean_latitude)
        logger.info("Cleaned latitude values")

        # Compute geodesic distance
        df_feat_eng['Geodesic Distance'] = df_feat_eng.apply(compute_geodesic_distance, axis=1)
        logger.info("Computed geodesic distances")

        # Compute Haversine distance
        df_feat_eng['Haversine Distance'] = df_feat_eng.apply(compute_haversine_distance, axis=1)
        logger.info("Computed Haversine distances")

        # Compute average speed
        df_feat_eng['Average Speed'] = df_feat_eng.apply(compute_driving_speed, axis=1)
        logger.info("Computed average speeds")

        # Count riders within radius
        riders_count = count_riders_within_radius(df_feat_eng)
        logger.info(f"Counted riders within {Config.RADIUS} km of accepted orders")

        return df_feat_eng, riders_count

    except Exception as e:
        logger.error(f"Error performing analysis: {str(e)}")
        raise

