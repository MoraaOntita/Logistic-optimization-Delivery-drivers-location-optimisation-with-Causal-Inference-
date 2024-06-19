# Variables
PYTHON = python3
SCRIPTS_DIR = scripts
CONFIG_DIR = $(SCRIPTS_DIR)/config
SQL_DIR = $(SCRIPTS_DIR)/sql_integration
MODELS_DIR = $(SCRIPTS_DIR)/models  # New models directory
INT_DIR = $(SCRIPTS_DIR)/int  # New int directory
TESTS_DIR = tests

# Targets
all: init_db load_data preprocess feature_engineering analysis test

init_db:
    psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f $(SQL_DIR)/init.sql

load_data:
    $(PYTHON) $(SCRIPTS_DIR)/load_data.py

preprocess:
    $(PYTHON) $(SCRIPTS_DIR)/data_preprocessing.py

feature_engineering:
    $(PYTHON) $(SCRIPTS_DIR)/feature_engineering.py

analysis:
    $(PYTHON) $(SCRIPTS_DIR)/analysis.py

# New target for models
train_models:
    $(PYTHON) $(MODELS_DIR)/train_models.py

# New target for integration scripts
run_int_scripts:
    $(PYTHON) $(INT_DIR)/script1.py
    $(PYTHON) $(INT_DIR)/script2.py
    # Add more scripts as needed

test:
    $(PYTHON) -m unittest discover -s $(TESTS_DIR) -p "test_*.py"

clean:
    # Optionally add commands to clean up temporary files or logs
    rm -rf logs/*   # Example: Clean up all files in the logs directory

.PHONY: all init_db load_data preprocess feature_engineering analysis train_models run_int_scripts test clean

