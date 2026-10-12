.PHONY: clean data features train predict visualize lint requirements

###########################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = machine-learning-project
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Build Features
features:
	$(PYTHON_INTERPRETER) src/features/build_features.py \
	data/processed/spam_clean.csv \
	data/processed/features

## Train Model
train:
	$(PYTHON_INTERPRETER) src/models/train_model.py \
	data/processed/features \
	models/spam_model.pkl

## Predict / Evaluate
predict:
	$(PYTHON_INTERPRETER) src/models/predict_model.py \
	models/spam_model.pkl \
	data/processed/features

## Visualize Results
visualize:
	$(PYTHON_INTERPRETER) src/visualization/visualize.py \
	models/spam_model.pkl \
	data/processed/features \
	reports/figures

## Delete all compiled Python files
clean:
	del /s /q *.pyc
	del /s /q __pycache__

## Lint using flake8
lint:
	flake8 src