# Variables
PYTHON = python3
SCRIPTS_DIR = scripts
CONFIG_DIR = $(SCRIPTS_DIR)/config

# Targets
all: preprocess feature_engineering analysis analysis_script

preprocess:
    $(PYTHON) $(SCRIPTS_DIR)/data_preprocessing.py

feature_engineering:
    $(PYTHON) $(SCRIPTS_DIR)/feat_eng.py

analysis:
    $(PYTHON) $(SCRIPTS_DIR)/main.py

analysis_script:
    $(PYTHON) $(SCRIPTS_DIR)/analysis.py

clean:
    rm -rf logs/*   # Clean up all files in the logs directory

.PHONY: all preprocess feature_engineering analysis analysis_script clean
