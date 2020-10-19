# run
run:
	poetry run python .\openhand_classifier

run37:
	python37 -m poetry run python .\openhand_classifier

dataset:
	python37 -m poetry run python .\openhand_classifier\scripts\DatasetBuilder.py

# formatting

fmt-black:
	poetry run black openhand_classifier/src/ tests/

# lint

lint-black:
	poetry run black --check openhand_classifier/src/ tests/

lint: lint-black

# test

test-pytest:
	poetry run pytest tests/

test: test-pytest
