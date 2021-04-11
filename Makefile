black:
	black clumper tests setup.py --check

flake:
	flake8 clumper tests setup.py

test:
	pytest

check: black flake test

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

install-test:
	python -m pip install -e ".[test]"
