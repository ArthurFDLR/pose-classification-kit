from ..imports.qt import QtWidgets, QtCore
from ..imports.tensorflow import tf, TF_LOADED
from ..imports.openpose import OPENPOSE_LOADED

if OPENPOSE_LOADED:
    from ..imports.openpose import op
from .dynamic_bar_graph_widget import BarGraphWidget
from .classifier_selection_widget import ClassifierSelectionWidget
from ...datasets.body_models import BODY18, BODY18_FLAT, BODY25, BODY25_FLAT, BODY25_to_BODY18_indices

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import figure, lines
from matplotlib.pyplot import cm


class BodyPlotWidget(QtWidgets.QWidget):
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

        numPartPairs = len(BODY25.pairs)
        color_map = cm.get_cmap("hsv", numPartPairs)
        self.pairLines = [
            lines.Line2D([], [], color=color_map(i)) for i in range(numPartPairs)
        ]

        for line in self.pairLines:
            self.ax.add_line(line)

    def plotBody(self, bodyKeypoints, accuracy: int):
        if self.isBodyData(bodyKeypoints):
            for i, line in enumerate(self.pairLines):
                keypoints_1, keypoints_2 = BODY25.pairs[i]
                if (
                    bodyKeypoints[2][keypoints_1] == 0.0
                    or bodyKeypoints[2][keypoints_2] == 0.0
                ):
                    line.set_data([], [])
                else:
                    line.set_data(
                        [bodyKeypoints[0][keypoints_1], bodyKeypoints[0][keypoints_2]],
                        [bodyKeypoints[1][keypoints_1], bodyKeypoints[1][keypoints_2]],
                    )
                # line.set_data(list(data[i][0]), list(data[i][1]))

            self.ax.set_title(
                "Accuracy: " + str(accuracy), fontsize=12, color="#454545"
            )
        else:
            self.clear()
            self.ax.set_title("")
        self.canvas.draw()

    def clear(self):
        for line in self.pairLines:
            line.set_data([], [])
        self.canvas.draw()

    def isBodyData(self, keypoints):
        if type(keypoints) == np.ndarray:
            if keypoints.shape == (3, 25):
                return True
        return False


class BodyAnalysisWidget(QtWidgets.QGroupBox):
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
    """

    def __init__(self):
        super().__init__(("Full body"))
        self.setStyleSheet(self.stylesheet)
        self.showInput = True
        self.classOutputs = []
        self.modelClassifier = None
        self.currentBodyModel = None
        self.currentPrediction = ""

        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)

        self.predictionLabel = QtWidgets.QLabel(self)
        self.predictionLabel.setObjectName("Large_Label")
        self.layout.addWidget(self.predictionLabel)
        self.predictionLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.classGraphWidget = BarGraphWidget()
        self.bodyGraphWidget = BodyPlotWidget()

        self.graphSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.graphSplitter.setChildrenCollapsible(False)
        self.graphSplitter.addWidget(self.bodyGraphWidget)
        self.graphSplitter.addWidget(self.classGraphWidget)
        self.graphSplitter.setStretchFactor(0, 2)
        self.graphSplitter.setStretchFactor(1, 1)

        self.layout.addWidget(self.graphSplitter)
        self.layout.setStretch(0, 0)
        self.layout.setStretch(1, 1)

    def setClassifierModel(self, model, classOutputs):  # model:tf.keras.models
        self.modelClassifier = model
        if model != None:
            self.modelInputShape = model.layers[0].input_shape[1:]
            if self.modelInputShape[0] == 25:
                self.currentBodyModel = BODY25
            elif self.modelInputShape[0] == 18:
                self.currentBodyModel = BODY18
            elif self.modelInputShape[0] == 50:
                self.currentBodyModel = BODY25_FLAT
            elif self.modelInputShape[0] == 36:
                self.currentBodyModel = BODY18_FLAT
            else:
                self.currentBodyModel = None
        else:
            self.modelInputShape = None
            self.currentBodyModel = None
        print(self.modelInputShape)
        self.classOutputs = classOutputs
        self.classGraphWidget.changeCategories(self.classOutputs)

    def drawBody(self, bodyKeypoints: np.ndarray, accuracy: float):
        """Draw keypoints of a body pose in the widget if showInput==True.

        Args:
            keypoints (np.ndarray((3,25),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        """
        if self.showInput:
            # self.bodyGraphWidget.setTitle('Detection accuracy: ' + str(accuracy))
            self.updatePredictedClass(bodyKeypoints)
            self.bodyGraphWidget.plotBody(bodyKeypoints, accuracy)

    def updatePredictedClass(self, keypoints: np.ndarray):
        """Draw keypoints of a body pose in the widget.

        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
        """

        prediction = [0 for i in self.classOutputs]
        title = ""
        if type(keypoints) != type(None):
            if self.modelClassifier is not None:
                
                if self.currentBodyModel == BODY25:
                    inputData = keypoints[:2].T
                elif self.currentBodyModel == BODY25_FLAT:
                    inputData = np.concatenate(keypoints[:2].T, axis=0)
                elif self.currentBodyModel == BODY18:
                    inputData = keypoints.T[BODY25_to_BODY18_indices][:,:2]
                elif self.currentBodyModel == BODY18_FLAT:
                    inputData = np.concatenate(keypoints.T[BODY25_to_BODY18_indices][:,:2], axis=0)
                
                prediction = self.modelClassifier.predict(np.array([inputData]))[0]
                self.currentPrediction = self.classOutputs[np.argmax(prediction)]
                title = self.currentPrediction

        self.classGraphWidget.updateValues(np.array(prediction))
        self.setPredictionText(title)

    def newModelLoaded(self, urlModel: str, modelInfo: dict, bodyID: int):
        if TF_LOADED:
            if urlModel == "None":
                self.setClassifierModel(None, [])
            else:
                if bodyID == 2:  # Check if classifier for body poses (not hands)
                    model = tf.keras.models.load_model(urlModel)
                    nbrClass = model.layers[-1].output_shape[1]
                    if modelInfo and modelInfo.get('labels') and len(modelInfo.get('labels')) == nbrClass:
                        classOutputs = modelInfo.get('labels')
                    else:
                        classOutputs = [str(i) for i in range(1,nbrClass+1)]
                    self.setClassifierModel(model, classOutputs)

    def getCurrentPrediction(self) -> str:
        return self.currentPrediction

    def setPredictionText(self, prediction: str):
        self.predictionLabel.setText(prediction)


class BodyClassifierWidget(QtWidgets.QWidget):
    stylesheet = """
    #Body_classifier {
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
        self.setObjectName("Body_classifier")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(self.stylesheet)

        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(10)
        effect.setOffset(0, 0)
        effect.setColor(QtCore.Qt.gray)
        self.setGraphicsEffect(effect)

        ## Structure
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)

        self.classifierWidget = ClassifierSelectionWidget(
            parent=self, bodyClassification=True
        )
        self.bodyAnalysis = BodyAnalysisWidget()
        self.classifierWidget.newClassifierModel_Signal.connect(
            self.bodyAnalysis.newModelLoaded
        )

        self.layout.addWidget(self.bodyAnalysis)
        self.layout.addWidget(self.classifierWidget)
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 0)
