import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from .preprocessing import preprocess_data
from ..config import Config
import logging

logging.basicConfig(filename=Config.LOG_FILE_PATH, level=logging.INFO)

def train_model():
    try:
        # Load data
        df = pd.read_csv(Config.DATA_FILE_PATH)
        
        # Preprocess data
        X, y = preprocess_data(df)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Base Random Forest classifier
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Parameter grid for Randomized Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Randomized Search Cross-Validation
        random_search_rf = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_grid,
            n_iter=10,  # Adjust as needed
            scoring='accuracy',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        # Fit the random search model
        random_search_rf.fit(X_train, y_train)

        # Best parameters found
        best_params_rf = random_search_rf.best_params_
        logging.info(f"Best Parameters for Random Forest: {best_params_rf}")

        # Evaluate on test set
        y_pred_rf = random_search_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        logging.info(f'Test Accuracy for Random Forest: {accuracy_rf:.2f}')

        # Classification report
        logging.info('\nClassification Report for Random Forest:')
        logging.info(classification_report(y_test, y_pred_rf))

        # Confusion matrix
        logging.info('\nConfusion Matrix for Random Forest:')
        logging.info(confusion_matrix(y_test, y_pred_rf))

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise RuntimeError(f"Error in training model: {e}")

