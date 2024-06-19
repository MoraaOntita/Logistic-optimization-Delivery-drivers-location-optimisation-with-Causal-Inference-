import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from io import StringIO

# Import the functions to be tested
from scripts.sql_intergration.load_data import load_csv_to_db, main


class TestLoadData(unittest.TestCase):

    @patch("scripts.sql_intergration.load_data.pd.read_csv")
    @patch("scripts.sql_intergration.load_data.psycopg2.connect")
    def test_load_csv_to_db(self, mock_connect, mock_read_csv):
        # Mock the DataFrame returned by pd.read_csv
        mock_df = pd.DataFrame({
            "column1": [1, 2],
            "column2": ["a", "b"]
        })
        mock_read_csv.return_value = mock_df

        # Mock the connection and cursor
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value

        # Call the function
        load_csv_to_db("dummy_conn", "dummy.csv", "dummy_table")

        # Assertions
        mock_read_csv.assert_called_once_with("dummy.csv")
        mock_connect.assert_called_once()
        mock_conn.cursor.assert_called_once()
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("scripts.sql_intergration.load_data.load_csv_to_db")
    def test_main(self, mock_load_csv_to_db):
        # Mock the load_csv_to_db function
        mock_load_csv_to_db.side_effect = [None, None]

        # Call the main function
        with patch('builtins.print') as mocked_print:
            main()

            # Assertions
            mock_load_csv_to_db.assert_any_call("path_to_csv_file_1.csv", "table_name_1")
            mock_load_csv_to_db.assert_any_call("path_to_csv_file_2.csv", "table_name_2")
            self.assertEqual(mock_load_csv_to_db.call_count, 2)

            # Check the print statements
            mocked_print.assert_any_call("Data loaded from path_to_csv_file_1.csv into table table_name_1")
            mocked_print.assert_any_call("Data loaded from path_to_csv_file_2.csv into table table_name_2")


if __name__ == "__main__":
    unittest.main()
