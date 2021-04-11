black:
	black pytorch_gan_trainer tests setup.py --check

flake:
	flake8 pytorch_gan_trainer tests setup.py --max-line-length 89

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
