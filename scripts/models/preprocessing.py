import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ..config import Config

def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Encode categorical column using LabelEncoder."""
    try:
        label_encoder = LabelEncoder()
        df[f'{column}_encoded'] = label_encoder.fit_transform(df[column])
        return df
    except Exception as e:
        raise RuntimeError(f"Error encoding categorical column '{column}': {e}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data preprocessing tasks."""
    try:
        # Define feature columns
        feature_columns = ['id', 'Trip_ID', 'driver_id', 'Hour', 'Start_Hour', 'Geodesic_Distance',
                           'Haversine_Distance', 'Average_Speed', 'Time_Since_Last_Trip',
                           'Origin-Destination_Count', 'Origin_Distance_to_City_Center',
                           'Destination_Distance_to_City_Center', 'Driver_Experience', 'Avg_Speed',
                           'Day_of_Week_Encoded', 'Previous_Trip_End_Hour',
                           'Previous_Trip_End_Day_of_Week', 'Previous_Trip_End_Month', 'Origin-Destination_Encoded']
        
        # Perform feature encoding, scaling, or other preprocessing steps as needed
        df = encode_categorical(df, 'driver_action')
        # Add other preprocessing steps
        
        # Select specific feature columns
        X = df[feature_columns]
        y = df['driver_action_encoded']
        
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing data: {e}")
