# Detect the operating system
ifeq ($(OS),Windows_NT)
    # Windows-specific settings
    VENV_DIR := venv
    PYTHON := $(VENV_DIR)\Scripts\python.exe
    PIP := $(VENV_DIR)\Scripts\pip.exe
    MYPY := $(VENV_DIR)\Scripts\mypy.exe
    FLAKE8 := $(VENV_DIR)\Scripts\flake8.exe
    RM := rmdir /S /Q
    DEL_PYC := del /S /Q *.pyc
    DEL_DIR := for /d %%x in (__pycache__) do $(RM) %%x
    DEL_MYPY_CACHE := for /d %%x in (.mypy_cache) do $(RM) %%x
    MAKE_PYTHON_COMMAND := python
    MAKE_CMD := cmd /c
    VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
else
    # Unix-like system settings
    VENV_DIR := venv
    PYTHON := $(VENV_DIR)/bin/python
    PIP := $(VENV_DIR)/bin/pip
    MYPY := $(VENV_DIR)/bin/mypy
    FLAKE8 := $(VENV_DIR)/bin/flake8
    RM := rm -rf
    DEL_PYC := find . -name "*.pyc" -delete
    DEL_DIR := find . -type d -name "__pycache__" -exec rm -rf {} +
    DEL_MYPY_CACHE := find . -type d -name ".mypy_cache" -exec rm -rf {} +
    MAKE_PYTHON_COMMAND := python3
    MAKE_CMD :=
    VENV_PYTHON := $(VENV_DIR)/bin/python
endif

# Define the tests directory
TESTS := tests

# Default target
.PHONY: all help install run test

all: help

# Help target to display available commands
help:
	@echo "Bike Rental Demand Prediction Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  venv          Create a Python virtual environment"
	@echo "  install       Install project dependencies"
	@echo "  run           Execute the main script"
	@echo "  test          Run unit tests"
	@echo "  help          Show this help message"

# Create a virtual environment (only if not already created)
venv: $(VENV_PYTHON)

$(VENV_PYTHON):
	@echo "Creating virtual environment in $(VENV_DIR)/ ..."
	$(MAKE_PYTHON_COMMAND) -m venv $(VENV_DIR)
	@echo "✓ Virtual environment created."

# Install dependencies only if virtual environment exists
install: venv
	@echo "Upgrading pip..."
	$(MAKE_CMD) $(PYTHON) -m pip install --upgrade pip
	@echo "Installing dependencies from requirements.txt..."
	$(MAKE_CMD) $(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed."

# Run the main script
run: $(VENV_PYTHON)
	@echo "Running the main script..."
	$(PYTHON) src/main.py

# Run unit tests
test: $(VENV_PYTHON)
	@echo "Running unit tests..."
	$(PYTHON) -m unittest discover -s $(TESTS)