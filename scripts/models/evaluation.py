import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from typing import Tuple
import pandas as pd
from sklearn.base import ClassifierMixin
from ..config import Config
from .preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, dict]:
    try:
        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f'Test Accuracy: {accuracy:.2f}')

        # Classification report
        metrics_dict = classification_report(y_test, y_pred, output_dict=True)
        logging.info('\nClassification Report:')
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                logging.info(f"Class: {key}")
                for k, v in value.items():
                    logging.info(f"{k}: {v}")
            else:
                logging.info(f"{key}: {value}")

        # Confusion matrix
        logging.info('\nConfusion Matrix:')
        cm = confusion_matrix(y_test, y_pred)
        logging.info(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')  # Save the plot as PNG file
        plt.show()

        return accuracy, metrics_dict

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise RuntimeError(f"Error in model evaluation: {e}")

def main():
    try:
        setup_logging()

        # Example usage in main function (optional)
        # Load data (example)
        df = pd.read_csv(Config.DATA_FILE_PATH)
        
        # Preprocess data (example)
        X, y = preprocess_data(df)
        
        # Split data into training and testing sets (example)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Example: Initialize and train a model (Random Forest)
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy, metrics_dict = evaluate_model(model, X_test, y_test)
        logging.info(f"Test Accuracy: {accuracy:.2f}")
        logging.info(f"Metrics Dictionary: {metrics_dict}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise RuntimeError(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
