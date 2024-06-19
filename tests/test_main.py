import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from scripts.main import (
    execute_sql_script, load_csv_to_db, main, with_postgres_connection
)
from scripts.feat_eng import preprocess_data, perform_feature_engineering
from scripts.analysis import perform_analysis
from scripts.config.config import Config

class TestMain(unittest.TestCase):

    def setUp(self):
        # Setup sample data for testing
        self.df1 = pd.DataFrame({
            'order_id': [1, 2, 3],
            'lat': [0.0, 1.0, 2.0],
            'lng': [0.0, 1.0, 2.0]
        })
        self.df2 = pd.DataFrame({
            'Trip ID': [1, 2, 3],
            'Trip Start Time': ['2023-01-01 08:00:00', '2023-01-02 12:00:00', '2023-01-03 18:00:00'],
            'Trip End Time': ['2023-01-01 09:00:00', '2023-01-02 13:00:00', '2023-01-03 19:00:00'],
            'Trip Origin': ['0.0,0.0', '1.0,1.0', '2.0,2.0'],
            'Trip Destination': ['1.0,1.0', '2.0,2.0', '3.0,3.0']
        })

    @patch('scripts.main.psycopg2.connect')
    def test_execute_sql_script(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        execute_sql_script('scripts/sql_integration/init.sql')

        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch('scripts.main.psycopg2.connect')
    def test_load_csv_to_db(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        load_csv_to_db(mock_conn, 'test.csv', 'test_table')

        mock_cursor.executemany.assert_called()
        mock_conn.commit.assert_called()

    @patch('scripts.main.setup_logging')
    @patch('scripts.main.preprocess_data')
    @patch('scripts.main.perform_feature_engineering')
    @patch('scripts.main.perform_analysis')
    @patch('scripts.main.pd.read_csv')
    def test_main_data_preprocessing_and_analysis(self, mock_read_csv, mock_perform_analysis, mock_perform_feature_engineering, mock_preprocess_data, mock_setup_logging):
        mock_setup_logging.return_value = None
        mock_read_csv.side_effect = [self.df1, self.df2]
        mock_preprocess_data.return_value = (self.df1, self.df2)
        mock_perform_feature_engineering.return_value = self.df2
        mock_perform_analysis.return_value = (pd.DataFrame(), 10)

        with patch('scripts.main.logging.info') as mock_log_info:
            main()

            mock_log_info.assert_any_call("Data preprocessing completed successfully.")
            mock_log_info.assert_any_call("Feature engineering completed successfully.")
            mock_log_info.assert_any_call("Analysis completed successfully.")
            mock_log_info.assert_any_call("Number of riders within 5 km of accepted orders: 10")

    @patch('scripts.main.logging.error')
    @patch('scripts.main.setup_logging')
    @patch('scripts.main.preprocess_data')
    @patch('scripts.main.perform_feature_engineering')
    @patch('scripts.main.perform_analysis')
    @patch('scripts.main.pd.read_csv')
    def test_main_data_preprocessing_and_analysis_failure(self, mock_read_csv, mock_perform_analysis, mock_perform_feature_engineering, mock_preprocess_data, mock_setup_logging, mock_log_error):
        mock_setup_logging.return_value = None
        mock_read_csv.side_effect = [self.df1, self.df2]
        mock_preprocess_data.side_effect = Exception('Preprocessing error')
        
        with self.assertRaises(Exception):
            main()
            mock_log_error.assert_called_with("An error occurred in main: Preprocessing error")

if __name__ == "__main__":
    unittest.main()
