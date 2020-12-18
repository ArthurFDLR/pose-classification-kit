# run
run:
	poetry run python .\openhand_app

dataset-csv:
	python -m poetry run python .\openhand_app\scripts\dataset_export.py

video-overlay:
	python -m poetry run python .\openhand_app\scripts\video_creation.py

# formatting

fmt-black:
	poetry run black openhand_app/

lint-black:
	poetry run black --check openhand_app/src/

lint: lint-black
