# run
run:
	poetry run python .\openhand_classifier

run37:
	python37 -m poetry run python .\openhand_classifier

dataset-csv:
	python37 -m poetry run python .\openhand_classifier\scripts\DatasetBuilder.py

video_overlay:
	python37 -m poetry run python .\openhand_classifier\scripts\video_creation.py

# formatting

fmt-black:
	poetry run black openhand_classifier/

# lint

lint-black:
	poetry run black --check openhand_classifier/src/

lint: lint-black

# test

#test-pytest:
#	poetry run pytest tests/

test: test-pytest

test-video:
	poetry run python .\openhand_classifier\src\video_widget.py
