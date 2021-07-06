import pathlib

# Path to OpenPose installation folder on your system.
OPENPOSE_PATH = pathlib.Path("C:/") / "Program files" / "OpenPose"

# Path to model folder.
MODELS_PATH = pathlib.Path(".").resolve() / "pose_classification_kit" / "models"

# Path to datasets folder.
DATASETS_PATH = pathlib.Path(".").resolve() / "pose_classification_kit" / "datasets"
