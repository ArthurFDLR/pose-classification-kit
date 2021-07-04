from .qt import QtWidgets, QtCore, pyqtSignal

import os
from __init__ import MODELS_PATH

class ClassifierSelectionWidget(QtWidgets.QWidget):
    # newClassifierModel_Signal: url to load classifier model, output labels, handID
    newClassifierModel_Signal = pyqtSignal(str, list, int)

    def __init__(self, parent=None, bodyClassification:bool=False):
        super().__init__()
        self.parent = parent
        self.modelRight = None
        self.modelLeft = None
        self.modelsPath = MODELS_PATH / ("Body" if bodyClassification else "Hands")
        self.bodyClassification = bodyClassification

        self.classOutputs = []
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
                urlClass = pathFolder / "class.txt"
                if urlClass.is_file():
                    with open(urlClass, "r") as file:
                        first_line = file.readline()
                    self.classOutputs = first_line.split(",")
                    print("Class model loaded.")

                if self.bodyClassification:
                    pathBody = pathFolder / (name + "_body.h5")
                    if pathBody.is_file():
                        self.newClassifierModel_Signal.emit(str(pathBody), self.classOutputs, 2)
                        print("Body model loaded.")
                    else:
                        self.newClassifierModel_Signal.emit("None", [], 2)
                else:
                    pathRight = pathFolder / (name + "_right.h5")
                    if pathRight.is_file():
                        self.newClassifierModel_Signal.emit(str(pathRight), self.classOutputs, 1)
                        print("Right hand model loaded.")
                    else:
                        self.newClassifierModel_Signal.emit("None", [], 1)
                    
                    pathLeft = pathFolder /  (name + "_left.h5")
                    if pathLeft.is_file():
                        self.newClassifierModel_Signal.emit(str(pathLeft), self.classOutputs, 0)
                        print("Left hand model loaded.")
                    else:
                        self.newClassifierModel_Signal.emit("None", [], 0)
        else:
            print("None")
            self.modelRight = None
            self.modelLeft = None
            self.classOutputs = []
            self.newClassifierModel_Signal.emit("None", [], -1)

    def getAvailableClassifiers(self):
        listOut = ["None"]
        listOut += [
            name
            for name in os.listdir(str(self.modelsPath))
            if (self.modelsPath / name).is_dir()
        ]
        return listOut

    def updateClassifier(self):
        self.classifierSelector.clear()
        self.classifierSelector.addItems(self.getAvailableClassifiers())
