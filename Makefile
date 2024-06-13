# Variables
PYTHON = python3
SCRIPTS_DIR = scripts
CONFIG_DIR = $(SCRIPTS_DIR)/config

# Targets
all: preprocess

preprocess:
	$(PYTHON) $(SCRIPTS_DIR)/data_preprocessing.py

clean:
	# Optionally add commands to clean up temporary files or logs

.PHONY: all preprocess clean
