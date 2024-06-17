# Variables
PYTHON = python3
SCRIPTS_DIR = scripts
CONFIG_DIR = $(SCRIPTS_DIR)/config
SQL_DIR = $(SCRIPTS_DIR)/sql_integration

# Targets
all: init_db load_data preprocess feature_engineering analysis

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

clean:
    # Optionally add commands to clean up temporary files or logs
    rm -rf logs/*   # Example: Clean up all files in the logs directory

.PHONY: all init_db load_data preprocess feature_engineering analysis clean
