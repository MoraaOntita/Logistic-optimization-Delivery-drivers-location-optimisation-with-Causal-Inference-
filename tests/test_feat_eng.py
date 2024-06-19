import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Import the functions to be tested
from scripts.feat_eng import (
    preprocess_datetime, extract_day_of_week, categorize_time_of_day, extract_hour_and_time_of_day,
    create_is_holiday_feature, preprocess_trip_times, split_origin_destination,
    extract_additional_time_features, calculate_trip_duration, merge_and_calculate_distances,
    scale_features
)


class TestFeatEng(unittest.TestCase):

    def setUp(self):
        # Set up a sample dataframe for testing
        self.df = pd.DataFrame({
            'Trip Start Time': ['2021-01-01 10:00:00', '2021-01-02 15:00:00'],
            'Trip End Time': ['2021-01-01 11:00:00', '2021-01-02 16:00:00'],
            'Trip Origin': ['0,0', '1,1'],
            'Trip Destination': ['2,2', '3,3'],
            'lat': [0, 1],
            'lng': [0, 1],
            'order_id': [1, 2]
        })

    @patch('scripts.feat_eng.logging.info')
    def test_preprocess_datetime(self, mock_logging_info):
        df_result = preprocess_datetime(self.df.copy())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_result['Trip Start Time']))
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_extract_day_of_week(self, mock_logging_info):
        df_processed = preprocess_datetime(self.df.copy())
        df_result = extract_day_of_week(df_processed)
        self.assertIn('Day of Week', df_result.columns)
        self.assertEqual(df_result['Day of Week'][0], 'Friday')
        mock_logging_info.assert_called_once()

    def test_categorize_time_of_day(self):
        self.assertEqual(categorize_time_of_day(8), 'Morning')
        self.assertEqual(categorize_time_of_day(13), 'Afternoon')
        self.assertEqual(categorize_time_of_day(18), 'Evening')
        self.assertEqual(categorize_time_of_day(22), 'Night')

    @patch('scripts.feat_eng.logging.info')
    def test_extract_hour_and_time_of_day(self, mock_logging_info):
        df_processed = preprocess_datetime(self.df.copy())
        df_result = extract_hour_and_time_of_day(df_processed)
        self.assertIn('Hour', df_result.columns)
        self.assertIn('Time of Day', df_result.columns)
        self.assertEqual(df_result['Hour'][0], 10)
        self.assertEqual(df_result['Time of Day'][0], 'Morning')
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_create_is_holiday_feature(self, mock_logging_info):
        with patch('scripts.feat_eng.Config.HOLIDAYS_2021', ['2021-01-01']), patch('scripts.feat_eng.Config.HOLIDAYS_2022', []):
            df_processed = preprocess_datetime(self.df.copy())
            df_result = create_is_holiday_feature(df_processed)
            self.assertIn('Is Holiday', df_result.columns)
            self.assertTrue(df_result['Is Holiday'][0])
            mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_preprocess_trip_times(self, mock_logging_info):
        df_result = preprocess_trip_times(self.df.copy())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_result['Trip Start Time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_result['Trip End Time']))
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_split_origin_destination(self, mock_logging_info):
        df_result = split_origin_destination(self.df.copy())
        self.assertIn('Origin Lat', df_result.columns)
        self.assertIn('Origin Lng', df_result.columns)
        self.assertIn('Destination Lat', df_result.columns)
        self.assertIn('Destination Lng', df_result.columns)
        self.assertEqual(df_result['Origin Lat'][0], 0.0)
        self.assertEqual(df_result['Destination Lat'][0], 2.0)
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_extract_additional_time_features(self, mock_logging_info):
        df_processed = preprocess_trip_times(self.df.copy())
        df_result = extract_additional_time_features(df_processed)
        self.assertIn('Start Hour', df_result.columns)
        self.assertIn('Start Day of Week', df_result.columns)
        self.assertEqual(df_result['Start Hour'][0], 10)
        self.assertEqual(df_result['Start Day of Week'][0], 4)
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_calculate_trip_duration(self, mock_logging_info):
        df_processed = preprocess_trip_times(self.df.copy())
        df_result = calculate_trip_duration(df_processed)
        self.assertIn('Trip Duration', df_result.columns)
        self.assertEqual(df_result['Trip Duration'][0], 60.0)
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    def test_merge_and_calculate_distances(self, mock_logging_info):
        df1 = pd.DataFrame({'order_id': [1, 2], 'lat': [0, 1], 'lng': [0, 1]})
        df2 = self.df.copy()
        df_result = merge_and_calculate_distances(df1, df2)
        self.assertIn('Driver Distance to Origin', df_result.columns)
        self.assertIn('Trip Distance', df_result.columns)
        self.assertAlmostEqual(df_result['Driver Distance to Origin'][0], 0.0, places=2)
        self.assertAlmostEqual(df_result['Trip Distance'][0], 314.40, places=2)  # Geodesic distance from (0,0) to (2,2)
        mock_logging_info.assert_called_once()

    @patch('scripts.feat_eng.logging.info')
    @patch('scripts.feat_eng.StandardScaler')
    def test_scale_features(self, mock_standard_scaler, mock_logging_info):
        mock_scaler_instance = MagicMock()
        mock_standard_scaler.return_value = mock_scaler_instance
        df_result = scale_features(self.df.copy())
        self.assertTrue(mock_scaler_instance.fit_transform.called)
        mock_logging_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
