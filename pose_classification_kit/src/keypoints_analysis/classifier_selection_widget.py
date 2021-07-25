from pose_classification_kit.datasets.body_models import BODY18
from ..imports.qt import QtWidgets, QtCore, pyqtSignal
from ...config import MODELS_PATH

import json

class ClassifierSelectionWidget(QtWidgets.QWidget):
    # newClassifierModel_Signal: url to load classifier model, model infos from JSON, handID
    newClassifierModel_Signal = pyqtSignal(str, object, int)

    def __init__(self, parent=None, bodyClassification: bool = False):
        super().__init__()
        self.parent = parent
        self.modelRight = None
        self.modelLeft = None
        self.modelsPath = MODELS_PATH / ("Body" if bodyClassification else "Hands")
        self.bodyClassification = bodyClassification

        #self.classOutputs = []
        self.leftWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        classifierLabel = QtWidgets.QLabel("Classifier:")
        classifierLabel.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.layout.addWidget(classifierLabel, 1, 0, 1, 1)

        self.classifierSelector = QtWidgets.QComboBox(
            cursor=QtCore.Qt.PointingHandCursor
        )
        self.classifierSelector.addItems(self.getAvailableClassifiers())
        self.layout.addWidget(self.classifierSelector, 1, 1, 1, 1)
        self.classifierSelector.currentTextChanged.connect(self.loadModel)

        updateClassifierButton = QtWidgets.QPushButton(
            "Update list",
            cursor=QtCore.Qt.PointingHandCursor,
            toolTip="Refresh list of model available in associated folder",
        )
        updateClassifierButton.clicked.connect(self.updateClassifier)
        self.layout.addWidget(updateClassifierButton, 1, 2, 1, 1)

    def loadModel(self, name: str):
        """Load full (structures + weigths) h5 model.

        Args:
            name (string): Name of the model. The folder .\models\name must contain: modelName_right.h5, modelName_left.h5, class.txt
        """
        if name != "None":
            pathFolder = self.modelsPath / name
            print(pathFolder)
            if pathFolder.is_dir():
                
                ModelInfoPath = next(pathFolder.glob("*.json"), None)
                modelInfo = None
                if ModelInfoPath:
                    with open(ModelInfoPath, "r") as f:
                        try:
                            modelInfo = json.load(f)
                        except:
                            modelInfo = None
                    #self.classOutputs = first_line.split(",")
                    #print("Class model loaded:", self.classOutputs)

                if self.bodyClassification:
                    availableModelPath = next(pathFolder.glob("*.h5"), None)
                    if availableModelPath:
                        self.newClassifierModel_Signal.emit(
                            str(availableModelPath), modelInfo, 2
                        )
                        print(str(availableModelPath), "loaded.")
                    else:
                        print(
                            "No model found."
                        )
                        self.newClassifierModel_Signal.emit("None", {}, 2)
                
                else:
                    availableModels = list(pathFolder.glob("*_right.h5"))
                    if len(availableModels) > 0:
                        self.newClassifierModel_Signal.emit(
                            str(availableModels[0]), modelInfo, 1
                        )
                        print("Right hand model loaded.")
                    else:
                        print("No right hand model found.")
                        self.newClassifierModel_Signal.emit("None", {}, 1)

                    availableModels = list(pathFolder.glob("*_left.h5"))
                    if len(availableModels) > 0:
                        self.newClassifierModel_Signal.emit(
                            str(availableModels[0]), modelInfo, 0
                        )
                        print("Left hand model loaded.")
                    else:
                        print("No left hand model found.")
                        self.newClassifierModel_Signal.emit("None", {}, 0)

        else:
            print("None")
#            self.modelRight = None
#            self.modelLeft = None
#            self.classOutputs = []
            self.newClassifierModel_Signal.emit("None", {}, -1)

    def getAvailableClassifiers(self):
        listOut = ["None"]
        # Get all directory that contains an h5 file.
        listOut += [x.stem for x in self.modelsPath.glob('*') if x.is_dir() and next(x.glob('*.h5'), None)]
        return listOut

    def updateClassifier(self):
        self.classifierSelector.clear()
        self.classifierSelector.addItems(self.getAvailableClassifiers())
