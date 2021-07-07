from pathlib import Path
from ..config import MODELS_PATH
from ..src.imports.tensorflow import tf

availableModelsPath = {
    "9class_3x64_body18":       MODELS_PATH / "Body" / "9Classes_3x64_BODY18" / "9Classes_3x64_body18.h5",
    "9class_3x64_body25":       MODELS_PATH / "Body" / "9Classes_3x64_BODY25" / "9Classes_3x64_body25.h5",
    "24class_2x128_rightHand":  MODELS_PATH / "Hands" / "24Output-2x128-17epochs" / "24Output-2x128-17epochs_right.h5",
    "24class_2x128_leftHand":   MODELS_PATH / "Hands" / "24Output-2x128-17epochs" / "24Output-2x128-17epochs_left.h5",
    "27class_3x64_rightHand":   MODELS_PATH / "Hands" / "27Class_3x64" / "27Class_3x64_right.h5",
    "27class_3x64_leftHand":    MODELS_PATH / "Hands" / "27Class_3x64" / "27Class_3x64_left.h5",
}

AVAILABLE_MODELS = list(availableModelsPath.keys())

for modelName, modelPath in availableModelsPath.items():
    if not modelPath.is_file():
        print(modelName, "missing at", modelPath)


def getModel(modelName: str):
    """ Load pre-trained available model -- see AVAILABLE_MODELS.

    Args:
        modelName (str): Name of the model to load.

    Returns:
        (tf.keras.Model, List[str]): Pre-trained Keras model and list of output class labels.
    """
    model = None
    labels = None

    if modelName in availableModelsPath:
        modelPath = availableModelsPath[modelName]
        model = tf.keras.models.load_model(modelPath)

        classPath = modelPath.parent / "class.txt"

        if classPath.is_file():
            with open(classPath, "r") as file:
                labels = file.readline().split(",")
        else:
            print("No class file available.")
    else:
        print("Model not found.")

    return model, labels
