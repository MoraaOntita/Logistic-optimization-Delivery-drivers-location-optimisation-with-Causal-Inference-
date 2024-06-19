import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import DATA_FILE, MLFLOW_SERVER_URI  # Import configuration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
    - file_path: Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into train and test sets.

    Args:
    - data: DataFrame containing the data.
    - target_column: Name of the target column.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random seed for reproducibility.

    Returns:
    - tuple: X_train, X_test, y_train, y_test
    """
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Data split into train and test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, max_depth: int = 5, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
    - X_train: Features for training.
    - y_train: Target labels for training.
    - n_estimators: Number of trees in the forest.
    - max_depth: Maximum depth of the trees.
    - random_state: Random seed for reproducibility.

    Returns:
    - RandomForestClassifier: Trained model.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate the trained model on test data.

    Args:
    - model: Trained model.
    - X_test: Features for testing.
    - y_test: Target labels for testing.

    Returns:
    - float: Accuracy score.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def log_mlflow_params(params: dict) -> None:
    """
    Log parameters to MLflow.

    Args:
    - params: Dictionary of parameters to log.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)
        logging.info(f"Logged parameter - {key}: {value}")

def log_mlflow_metrics(metrics: dict) -> None:
    """
    Log metrics to MLflow.

    Args:
    - metrics: Dictionary of metrics to log.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
        logging.info(f"Logged metric - {key}: {value}")

def log_mlflow_model(model: RandomForestClassifier, artifact_name: str = "random_forest_model") -> None:
    """
    Log the trained model as an artifact in MLflow.

    Args:
    - model: Trained model object.
    - artifact_name: Name to assign to the logged artifact.
    """
    try:
        mlflow.sklearn.log_model(model, artifact_name)
        logging.info(f"Logged model artifact as {artifact_name}")
    except Exception as e:
        logging.error(f"Error logging model artifact: {e}")
        raise

def main():
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        mlflow.start_run()

        # Load data
        data = load_data(DATA_FILE)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(data, "target")

        # Set parameters
        n_estimators = 100
        max_depth = 5

        # Log parameters
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        log_mlflow_params(params)

        # Train model
        model = train_model(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Log metrics
        metrics = {"accuracy": accuracy}
        log_mlflow_metrics(metrics)

        # Log model artifact
        log_mlflow_model(model)

        # End MLflow run
        mlflow.end_run()

        print("MLflow tracking completed successfully.")
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        mlflow.end_run()
        raise

if __name__ == "__main__":
    main()

