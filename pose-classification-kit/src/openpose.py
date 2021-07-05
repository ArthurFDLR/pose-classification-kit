import sys
import os

from ..config import OPENPOSE_PATH

try:
    sys.path.append(str(OPENPOSE_PATH / "build" / "python" / "openpose" / "Release"))
    releasePATH = OPENPOSE_PATH / "build" / "x64" / "Release"
    binPATH = OPENPOSE_PATH / "build" / "bin"
    OPENPOSE_MODELS_PATH = OPENPOSE_PATH / "models"
    os.environ["PATH"] = (
        os.environ["PATH"] + ";" + str(releasePATH) + ";" + str(binPATH) + ";"
    )
    import pyopenpose as op

    OPENPOSE_LOADED = True
except:
    OPENPOSE_LOADED = False
    print("OpenPose ({}) loading failed.".format(str(OPENPOSE_PATH)))
