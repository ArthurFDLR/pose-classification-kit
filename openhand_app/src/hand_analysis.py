from .qt import QtWidgets, QtCore, pyqtSignal
from .tensorflow import tf, TF_LOADED
from .dynamicBarGraphWidget import BarGraphWidget

import numpy as np
import os
from __init__ import MODELS_PATH

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import figure, lines


class HandPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = FigureCanvas(figure.Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        self.setMinimumHeight(50)

        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim([-1.0, 1.0])
        self.ax.set_ylim([-1.0, 1.0])
        self.ax.set_aspect("equal")

        self.fingerLines = [
            lines.Line2D([], [], color="r"),
            lines.Line2D([], [], color="y"),
            lines.Line2D([], [], color="g"),
            lines.Line2D([], [], color="b"),
            lines.Line2D([], [], color="m"),
        ]

        for line in self.fingerLines:
            self.ax.add_line(line)

    def plotHand(self, handKeypoints, accuracy: int):
        if self.isHandData(handKeypoints):
            colors = ["r", "y", "g", "b", "m"]
            data = [
                handKeypoints[:, 0:5],
                np.insert(handKeypoints[:, 5:9].T, 0, handKeypoints[:, 0], axis=0).T,
                np.insert(handKeypoints[:, 9:13].T, 0, handKeypoints[:, 0], axis=0).T,
                np.insert(handKeypoints[:, 13:17].T, 0, handKeypoints[:, 0], axis=0).T,
                np.insert(handKeypoints[:, 17:21].T, 0, handKeypoints[:, 0], axis=0).T,
            ]
            for i, line in enumerate(self.fingerLines):
                line.set_data(data[i][0], data[i][1])
            self.ax.set_title(
                "Accuracy: " + str(accuracy), fontsize=12, color="#454545"
            )
        else:
            self.clear()
            self.ax.set_title("")
        self.canvas.draw()

    def clear(self):
        for line in self.fingerLines:
            line.set_data([], [])
        self.canvas.draw()

    def isHandData(self, keypoints):
        b = False
        if type(keypoints) == np.ndarray:
            if keypoints.shape == (3, 21):
                b = True
        return b


class HandAnalysisWidget(QtWidgets.QGroupBox):
    stylesheet = """
    #Large_Label {
        font-size: 26px;
        color: #9500ff;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }

    QSplitter::handle {
        color: #cbcbcb;
        border: 1px solid #cbcbcb;
        border-radius: 2px;
    }

    QSplitter::handle:horizontal {
        height: 1px 
    }
    """

    def __init__(self, handID: int, showInput: bool = True):
        super().__init__(("Right" if handID == 1 else "Left") + " hand")
        self.setStyleSheet(self.stylesheet)
        self.handID = handID
        self.showInput = showInput
        self.classOutputs = []
        self.modelClassifier = None
        self.currentPrediction = ""

        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        self.predictionLabel = QtWidgets.QLabel(self)
        self.predictionLabel.setObjectName("Large_Label")
        self.layout.addWidget(self.predictionLabel)
        self.predictionLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.classGraphWidget = BarGraphWidget()

        if self.showInput:
            self.handGraphWidget = HandPlotWidget()
            self.graphSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.graphSplitter.setChildrenCollapsible(False)
            self.graphSplitter.addWidget(self.handGraphWidget)
            self.graphSplitter.addWidget(self.classGraphWidget)
            self.graphSplitter.setStretchFactor(0, 2)
            self.graphSplitter.setStretchFactor(1, 1)

            self.layout.addWidget(self.graphSplitter)
        else:
            self.layout.addWidget(self.classGraphWidget)

    def setClassifierModel(self, model, classOutputs):  # model:tf.keras.models
        self.modelClassifier = model
        self.classOutputs = classOutputs

    def drawHand(self, handKeypoints: np.ndarray, accuracy: float):
        """Draw keypoints of a hand pose in the widget if showInput==True.

        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        """
        if self.showInput:
            # self.handGraphWidget.setTitle('Detection accuracy: ' + str(accuracy))
            self.updatePredictedClass(handKeypoints)
            self.handGraphWidget.plotHand(handKeypoints, accuracy)

    def updatePredictedClass(self, keypoints: np.ndarray):
        """Draw keypoints of a hand pose in the widget.

        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
        """

        prediction = [0 for i in self.classOutputs]
        title = ""
        if type(keypoints) != type(None):
            inputData = []
            for i in range(keypoints.shape[1]):
                inputData.append(keypoints[0, i])  # add x
                inputData.append(keypoints[1, i])  # add y
            inputData = np.array(inputData)

            if self.modelClassifier is not None:
                prediction = self.modelClassifier.predict(np.array([inputData]))[0]
                self.currentPrediction = self.classOutputs[np.argmax(prediction)]
                title = self.currentPrediction

        self.classGraphWidget.updateValues(np.array(prediction))
        self.setPredictionText(title)

    def newModelLoaded(self, urlModel: str, classOutputs: list, handID: int):
        if TF_LOADED:
            if urlModel == "None":
                self.modelClassifier = None
                self.classOutputs = []
                self.classGraphWidget.changeCategories(self.classOutputs)
            else:
                if handID == self.handID:
                    self.modelClassifier = tf.keras.models.load_model(urlModel)
                    self.classOutputs = classOutputs
                    self.classGraphWidget.changeCategories(self.classOutputs)

    def getCurrentPrediction(self) -> str:
        return self.currentPrediction

    def setPredictionText(self, prediction: str):
        self.predictionLabel.setText(prediction)


class HandClassifierWidget(QtWidgets.QWidget):
    stylesheet = """
    #Hand_classifier {
        background-color: white;
        border-radius: 3px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    QGroupBox {
        font-size: 16px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    QLabel {
        font-size: 16px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    QPushButton {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: white;
        padding: 3px;
    }
    QComboBox {
        border: 1px solid #cbcbcb;
        border-radius: 3px;
        font-size: 16px;
        background: white;
    }
    QPushButton:hover {
        border-color: rgb(139, 173, 228);
    }
    QPushButton:pressed {
        background: #cbcbcb;
    }
    """

    def __init__(self):
        super().__init__()
        ## Widget style
        self.setObjectName("Hand_classifier")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(self.stylesheet)

        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(10)
        effect.setOffset(0, 0)
        effect.setColor(QtCore.Qt.gray)
        self.setGraphicsEffect(effect)

        ## Structure
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 0)

        self.classifierWidget = ClassifierSelectionWidget(self)
        self.layout.addWidget(self.classifierWidget, 2, 0, 1, 2)

        self.leftHandAnalysis = HandAnalysisWidget(0)
        self.classifierWidget.newClassifierModel_Signal.connect(
            self.leftHandAnalysis.newModelLoaded
        )
        self.layout.addWidget(self.leftHandAnalysis, 0, 0, 2, 1)

        self.rightHandAnalysis = HandAnalysisWidget(1)
        self.classifierWidget.newClassifierModel_Signal.connect(
            self.rightHandAnalysis.newModelLoaded
        )
        self.layout.addWidget(self.rightHandAnalysis, 0, 1, 2, 1)


class ClassifierSelectionWidget(QtWidgets.QWidget):
    # newClassifierModel_Signal: url to load classifier model, output labels, handID
    newClassifierModel_Signal = pyqtSignal(str, list, int)

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.modelRight = None
        self.modelLeft = None

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
            urlFolder = MODELS_PATH / name
            print(urlFolder)
            if urlFolder.is_dir():
                urlRight = urlFolder / (name + "_right.h5")
                urlLeft = urlFolder /  (name + "_left.h5")
                urlClass = urlFolder / "class.txt"
                if urlClass.is_file():
                    with open(urlClass, "r") as file:
                        first_line = file.readline()
                    self.classOutputs = first_line.split(",")
                    print("Class model loaded.")
                if urlRight.is_file():
                    self.newClassifierModel_Signal.emit(str(urlRight), self.classOutputs, 1)
                    print("Right hand model loaded.")
                else:
                    self.newClassifierModel_Signal.emit("None", [], 1)
                if urlLeft.is_file():
                    self.newClassifierModel_Signal.emit(str(urlLeft), self.classOutputs, 0)
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
            for name in os.listdir(str(MODELS_PATH))
            if (MODELS_PATH / name).is_dir()
        ]
        return listOut

    def updateClassifier(self):
        self.classifierSelector.clear()
        self.classifierSelector.addItems(self.getAvailableClassifiers())
