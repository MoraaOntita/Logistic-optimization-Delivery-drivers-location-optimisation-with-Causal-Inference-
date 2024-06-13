# Variables
PYTHON = python3
SCRIPTS_DIR = scripts
CONFIG_DIR = $(SCRIPTS_DIR)/config

# Targets
all: preprocess feature_engineering analysis

preprocess:
	$(PYTHON) $(SCRIPTS_DIR)/data_preprocessing.py

feature_engineering:
	$(PYTHON) $(SCRIPTS_DIR)/feat_eng.py

analysis:
	$(PYTHON) $(SCRIPTS_DIR)/analysis.py

clean:
	# Optionally add commands to clean up temporary files or logs
	rm -rf logs/*   # Example: Clean up all files in the logs directory

.PHONY: all preprocess feature_engineering analysis clean

