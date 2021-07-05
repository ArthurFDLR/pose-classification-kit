# formatting

fmt-black:
	poetry run black pose-classification-kit/

lint-black:
	poetry run black --check pose-classification-kit/src/

lint: lint-black
