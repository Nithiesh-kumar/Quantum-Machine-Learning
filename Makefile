# Makefile for Quantum ML System

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe
PYTEST := .venv/Scripts/pytest.exe

.PHONY: help install check test clean

help:
	@echo "Available commands:"
	@echo "  make install - Install requirements into virtualenv"
	@echo "  make check   - Verify environment and print package versions"
	@echo "  make test    - Run test suite with pytest"
	@echo "  make clean   - Remove caches and temporary files"

install:
	$(PIP) install -r requirements.txt

check:
	$(PYTHON) check_env.py

test:
	$(PYTEST) tests/ -v

clean:
	powershell -Command "Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force; Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force"
