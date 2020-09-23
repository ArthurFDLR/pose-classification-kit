# run
run:
	poetry run python .\openhand_classifier

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
