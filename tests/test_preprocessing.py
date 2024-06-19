import unittest
import pandas as pd
from scripts.models.preprocessing import preprocess_data
from scripts.config import Config

class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # Example: Load mock dataset for preprocessing
        self.df = pd.read_csv(Config.TEST_DATA_PATH)  # Adjust as needed

    def test_preprocess_data(self):
        X, y = preprocess_data(self.df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertFalse(X.empty)
        self.assertFalse(y.empty)
        # Add more specific assertions based on your preprocessing logic

if __name__ == '__main__':
    unittest.main()
