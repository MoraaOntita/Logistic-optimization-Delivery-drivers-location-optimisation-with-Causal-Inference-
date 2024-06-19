import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the functions to be tested
from scripts.analysis import clean_latitude, compute_geodesic_distance, compute_haversine_distance, compute_driving_speed, count_riders_within_radius, perform_analysis


class TestAnalysis(unittest.TestCase):

    def test_clean_latitude(self):
        self.assertEqual(clean_latitude(-100), -90)
        self.assertEqual(clean_latitude(100), 90)
        self.assertEqual(clean_latitude(45), 45)

    @patch('scripts.analysis.geodesic')
    def test_compute_geodesic_distance(self, mock_geodesic):
        mock_row = pd.Series({'Origin Lat': 0, 'Origin Lng': 0, 'Destination Lat': 0, 'Destination Lng': 0})
        mock_geodesic.return_value.kilometers = 100

        result = compute_geodesic_distance(mock_row)
        self.assertEqual(result, 100)
        mock_geodesic.assert_called_once()

    @patch('scripts.analysis.haversine')
    def test_compute_haversine_distance(self, mock_haversine):
        mock_row = pd.Series({'Origin Lat': 0, 'Origin Lng': 0, 'Destination Lat': 0, 'Destination Lng': 0})
        mock_haversine.return_value = 100

        result = compute_haversine_distance(mock_row)
        self.assertEqual(result, 100)
        mock_haversine.assert_called_once()

    def test_compute_driving_speed(self):
        mock_row = pd.Series({'Haversine Distance': 100, 'Trip Duration': 120})
        result = compute_driving_speed(mock_row)
        self.assertEqual(result, 50)

        mock_row_zero_duration = pd.Series({'Haversine Distance': 100, 'Trip Duration': 0})
        result_zero_duration = compute_driving_speed(mock_row_zero_duration)
        self.assertEqual(result_zero_duration, 0)

    @patch('scripts.analysis.haversine')
    def test_count_riders_within_radius(self, mock_haversine):
        mock_haversine.side_effect = lambda x, y: 0.4  # Distance within radius

        mock_df = pd.DataFrame({
            'lat': [1, 2, 3],
            'lng': [1, 2, 3],
            'driver_action': ['accepted', 'accepted', 'rejected']
        })

        result = count_riders_within_radius(mock_df)
        self.assertEqual(result, 6)  # 2 accepted riders with 3 riders each in the radius

    @patch('scripts.analysis.compute_geodesic_distance')
    @patch('scripts.analysis.compute_haversine_distance')
    @patch('scripts.analysis.compute_driving_speed')
    @patch('scripts.analysis.count_riders_within_radius')
    @patch('scripts.analysis.clean_latitude')
    def test_perform_analysis(self, mock_clean_latitude, mock_count_riders_within_radius, mock_compute_driving_speed, mock_compute_haversine_distance, mock_compute_geodesic_distance):
        mock_clean_latitude.side_effect = lambda x: x
        mock_compute_geodesic_distance.side_effect = lambda x: 100
        mock_compute_haversine_distance.side_effect = lambda x: 100
        mock_compute_driving_speed.side_effect = lambda x: 50
        mock_count_riders_within_radius.return_value = 2

        mock_df = pd.DataFrame({
            'Origin Lat': [0, 10],
            'Origin Lng': [0, 10],
            'Destination Lat': [20, 30],
            'Destination Lng': [20, 30],
            'Trip Duration': [60, 120],
            'driver_action': ['accepted', 'rejected'],
            'lat': [1, 2],
            'lng': [1, 2]
        })

        result_df, riders_count = perform_analysis(mock_df)

        self.assertEqual(riders_count, 2)
        self.assertTrue('Geodesic Distance' in result_df.columns)
        self.assertTrue('Haversine Distance' in result_df.columns)
        self.assertTrue('Average Speed' in result_df.columns)
        self.assertEqual(result_df['Geodesic Distance'][0], 100)
        self.assertEqual(result_df['Haversine Distance'][0], 100)
        self.assertEqual(result_df['Average Speed'][0], 50)


if __name__ == "__main__":
    unittest.main()
