import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the functions to be tested
from scripts.data_preprocessing import setup_logging, load_data, handle_missing_values, preprocess_data


class TestDataPreprocessing(unittest.TestCase):

    @patch('scripts.data_preprocessing.logging.basicConfig')
    def test_setup_logging(self, mock_basicConfig):
        setup_logging()
        mock_basicConfig.assert_called_once_with(
            filename='path_to_log_file.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @patch('scripts.data_preprocessing.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_df2 = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
        mock_read_csv.side_effect = [mock_df1, mock_df2]

        df1, df2 = load_data()
        self.assertTrue(mock_read_csv.call_count, 2)
        self.assertTrue(df1.equals(mock_df1))
        self.assertTrue(df2.equals(mock_df2))

    @patch('scripts.data_preprocessing.pd.read_csv')
    @patch('scripts.data_preprocessing.logging.error')
    def test_load_data_file_not_found(self, mock_logging_error, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        with self.assertRaises(FileNotFoundError):
            load_data()
        mock_logging_error.assert_called_once_with("File not found: File not found")

    @patch('scripts.data_preprocessing.pd.read_csv')
    @patch('scripts.data_preprocessing.logging.error')
    def test_load_data_empty_data_error(self, mock_logging_error, mock_read_csv):
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        with self.assertRaises(pd.errors.EmptyDataError):
            load_data()
        mock_logging_error.assert_called_once_with("No data: No data")

    def test_handle_missing_values(self):
        mock_df1 = pd.DataFrame({
            'col1': [1, 2],
            'drop_col': [3, 4]
        })
        mock_df2 = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [None, 'b', 'c']
        })

        processed_df1, processed_df2 = handle_missing_values(mock_df1, mock_df2)

        self.assertNotIn('drop_col', processed_df1.columns)
        self.assertEqual(processed_df2['col1'].isna().sum(), 0)
        self.assertEqual(processed_df2['col2'].isna().sum(), 0)

    @patch('scripts.data_preprocessing.load_data')
    @patch('scripts.data_preprocessing.handle_missing_values')
    def test_preprocess_data(self, mock_handle_missing_values, mock_load_data):
        mock_df1 = pd.DataFrame({'col1': [1, 2]})
        mock_df2 = pd.DataFrame({'col1': [3, 4]})
        mock_load_data.return_value = (mock_df1, mock_df2)
        mock_handle_missing_values.return_value = (mock_df1, mock_df2)

        df1, df2 = preprocess_data()

        mock_load_data.assert_called_once()
        mock_handle_missing_values.assert_called_once_with(mock_df1, mock_df2)
        self.assertTrue(df1.equals(mock_df1))
        self.assertTrue(df2.equals(mock_df2))


if __name__ == "__main__":
    unittest.main()
