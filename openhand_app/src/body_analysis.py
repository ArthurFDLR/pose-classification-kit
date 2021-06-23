from .qt import QtWidgets, QtCore
from .tensorflow import tf, TF_LOADED
from .dynamic_bar_graph_widget import BarGraphWidget
from .classifier_selection_widget import ClassifierSelectionWidget

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import figure, lines


class BodyPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = FigureCanvas(figure.Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        self.setMinimumHeight(50)

        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim([0.0, 400.0])
        self.ax.set_ylim([0.0, 400.0])
        self.ax.set_aspect("equal")

        self.fingerLines = [
            lines.Line2D([], [], color="r"),
            lines.Line2D([], [], color="y")
        ]

        for line in self.fingerLines:
            self.ax.add_line(line)

    def plotBody(self, bodyKeypoints, accuracy: int):
        if self.isBodyData(bodyKeypoints):
            data = [
                [bodyKeypoints[0][0:2], bodyKeypoints[1][0:2]],
                [bodyKeypoints[0][1:3], bodyKeypoints[1][1:3]]
            ]
            for i, line in enumerate(self.fingerLines):
                line.set_data(list(data[i][0]), list(data[i][1]))
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

    def __init__(self, showInput: bool = True):
        super().__init__(("Full body"))
        self.setStyleSheet(self.stylesheet)
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
        self.bodyGraphWidget = BodyPlotWidget()

        self.graphSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.graphSplitter.setChildrenCollapsible(False)
        self.graphSplitter.addWidget(self.bodyGraphWidget)
        self.graphSplitter.addWidget(self.classGraphWidget)
        self.graphSplitter.setStretchFactor(0, 2)
        self.graphSplitter.setStretchFactor(1, 1)

        self.layout.addWidget(self.graphSplitter)

    def setClassifierModel(self, model, classOutputs):  # model:tf.keras.models
        self.modelClassifier = model
        self.classOutputs = classOutputs

    def drawBody(self, bodyKeypoints: np.ndarray, accuracy: float):
        """Draw keypoints of a body pose in the widget if showInput==True.

        Args:
            keypoints (np.ndarray((3,25),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        """
        if self.showInput:
            # self.bodyGraphWidget.setTitle('Detection accuracy: ' + str(accuracy))
            # self.updatePredictedClass(bodyKeypoints)
            self.bodyGraphWidget.plotBody(bodyKeypoints, accuracy)
            print(bodyKeypoints.shape if type(bodyKeypoints) != type(None) else '')
    
    '''
    def updatePredictedClass(self, keypoints: np.ndarray):
        """Draw keypoints of a body pose in the widget.

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

    def newModelLoaded(self, urlModel: str, classOutputs: list, bodyID: int):
        if TF_LOADED:
            if urlModel == "None":
                self.modelClassifier = None
                self.classOutputs = []
                self.classGraphWidget.changeCategories(self.classOutputs)
            else:
                if bodyID == self.bodyID:
                    self.modelClassifier = tf.keras.models.load_model(urlModel)
                    self.classOutputs = classOutputs
                    self.classGraphWidget.changeCategories(self.classOutputs)

    def getCurrentPrediction(self) -> str:
        return self.currentPrediction

    def setPredictionText(self, prediction: str):
        self.predictionLabel.setText(prediction)
    '''

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
        #self.layout.setRowStretch(0, 1)
        #self.layout.setRowStretch(1, 0)

        self.classifierWidget = ClassifierSelectionWidget(self)
        self.bodyAnalysis = BodyAnalysisWidget()
        #self.classifierWidget.newClassifierModel_Signal.connect(
        #    self.bodyAnalysis.newModelLoaded
        #)

        self.layout.addWidget(self.bodyAnalysis)
        self.layout.addWidget(self.classifierWidget)