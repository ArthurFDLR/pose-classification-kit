# run
run:
	poetry run python .\openhand_classifier\VideoAnalysis.py

# formatting

fmt-black:
	poetry run black beancount_n26/ tests/

# lint

lint-black:
	poetry run black --check beancount_n26/ tests/

lint-flake8:
	poetry run flake8 beancount_n26/ tests/

lint: lint-black lint-flake8

# test

test-pytest:
	poetry run pytest tests/

test: test-pytest
