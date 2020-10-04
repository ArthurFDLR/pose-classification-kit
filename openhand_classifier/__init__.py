__version__ = '0.1.0'

from src.VideoInput import *
from src.HandAnalysis import *
from src.PoseClassifier import *
from src.DatasetController import *

# Path to OpenPose installation folder on your system.
OPENPOSE_PATH = pathlib.Path("C:/") / "Program files" / "OpenPose"