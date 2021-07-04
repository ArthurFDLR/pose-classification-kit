# run
run:
	poetry run python .\Application

dataset-csv:
	poetry run python .\Application\scripts\dataset_export.py

video-overlay:
	poetry run python .\Application\scripts\video_creation.py

# formatting

fmt-black:
	poetry run black Application/

lint-black:
	poetry run black --check Application/src/

lint: lint-black
