from src.qt import QtWidgets, QtGui
from src.dataset_controller import DatasetControllerWidget
from src.video_manager import CameraInput, VideoViewerWidget
from src.hand_analysis import HandClassifierWidget, TF_STATUS_STR, TF_LOADED
from src.openpose_thread import VideoAnalysisThread, OPENPOSE_LOADED

from __init__ import OPENPOSE_PATH

import time
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        ## Init
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("OpenHand classifier")

        ## Parameters
        self.isRecording = False
        self.realTimeHandDraw = True
        self.parent = parent

        ## Widgets
        self.cameraInput = CameraInput()

        self.videoViewer = VideoViewerWidget(self.cameraInput.getAvailableCam())
        self.videoViewer.camera_selector.currentIndexChanged.connect(
            self.cameraInput.select_camera
        )
        self.videoViewer.refreshButton.clicked.connect(self.refreshCameraList)

        self.datasetController = DatasetControllerWidget(self)
        self.datasetController.realTimeHandDraw_Signal.connect(
            self.changeHandDrawingState
        )

        videoHeight = 480  # 480p
        self.AnalysisThread = VideoAnalysisThread(self.cameraInput)
        self.AnalysisThread.newPixmap.connect(self.videoViewer.setImage)
        self.AnalysisThread.newPixmap.connect(self.analyseNewImage)
        self.AnalysisThread.setResolutionStream(
            int(videoHeight * (16.0 / 9.0)), videoHeight
        )
        self.videoViewer.setVideoSize(int(videoHeight * (16.0 / 9.0)), videoHeight)

        self.AnalysisThread.start()
        self.AnalysisThread.setState(True)

        self.handClassifier = HandClassifierWidget()

        ## Structure
        mainWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(mainWidget)

        self.layout = QtWidgets.QGridLayout(mainWidget)
        mainWidget.setLayout(self.layout)
        self.layout.addWidget(self.handClassifier, 0, 1, 3, 1)
        self.layout.addWidget(self.videoViewer, 0, 0, 1, 1)
        self.layout.addWidget(self.datasetController, 1, 0, 1, 1)
        self.layout.setRowStretch(0, 0)
        self.layout.setRowStretch(1, 0)
        self.layout.setRowStretch(2, 1)
        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)

        ## Menu
        bar = self.menuBar()
        fileAction = bar.addMenu("Dataset")

        openAct = QtWidgets.QAction("&Open", self)
        openAct.setShortcut("Ctrl+O")
        openAct.setStatusTip("Open dataset")
        openAct.triggered.connect(self.datasetController.loadFile)
        fileAction.addAction(openAct)

        initAct = QtWidgets.QAction("&Create new ...", self)
        initAct.setShortcut("Ctrl+N")
        initAct.setStatusTip("Create dataset")
        initAct.triggered.connect(self.datasetController.createDataset)
        fileAction.addAction(initAct)

        saveAct = QtWidgets.QAction("&Save", self)
        saveAct.setShortcut("Ctrl+S")
        saveAct.setStatusTip("Save dataset")
        saveAct.triggered.connect(self.datasetController.writeDataToTxt)
        fileAction.addAction(saveAct)

        ## Status Bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.openpose_path = "OpenPose installation path: " + str(OPENPOSE_PATH)
        self.openpose_path_label = QtWidgets.QLabel(
            self.openpose_path,
            toolTip="If incorrect, change in ./openhand_classifier/src/__init__.py",
        )
        self.statusBar.addWidget(self.openpose_path_label)

        self.openposeStatus = (
            "OpenPose running." if OPENPOSE_LOADED else "OpenPose not found"
        )
        self.openposeStatusLabel = QtWidgets.QLabel(
            '<span style="color:'
            + ("green" if OPENPOSE_LOADED else "red")
            + '">'
            + self.openposeStatus
            + "</span>"
        )
        self.statusBar.addWidget(self.openposeStatusLabel)

        self.tfStatusLabel = QtWidgets.QLabel(
            '<span style="color:'
            + ("green" if TF_LOADED else "red")
            + '">'
            + TF_STATUS_STR
            + "</span>"
        )
        self.statusBar.addWidget(self.tfStatusLabel)

    def closeEvent(self, event):
        print("Closing")
        exitBool = True
        if self.datasetController.isSaved():
            exitBool = True
        else:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Hand pose classifier",
                "Do you want to save "
                + self.datasetController.getPoseName()
                + " dataset?",
                buttons=QtWidgets.QMessageBox.StandardButtons(
                    QtWidgets.QMessageBox.Yes
                    | QtWidgets.QMessageBox.No
                    | QtWidgets.QMessageBox.Cancel
                ),
            )

            if reply == QtWidgets.QMessageBox.Cancel:
                exitBool = False
            elif reply == QtWidgets.QMessageBox.No:
                exitBool = True
            elif reply == QtWidgets.QMessageBox.Yes:
                self.datasetController.writeDataToTxt()
                exitBool = True

        if exitBool:
            event.accept()
            self.AnalysisThread.terminate()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.key() == 16777223 and self.datasetController.deleteButton.isEnabled():
            self.datasetController.removeEntryDataset(
                self.datasetController.currentDataIndex
            )

    def refreshCameraList(self):
        camList = self.cameraInput.refreshCameraList()
        if not camList:
            print("No camera")
        else:
            self.videoViewer.camera_selector.clear()
            self.videoViewer.camera_selector.addItems(
                [c.description() for c in camList]
            )

    def analyseNewImage(self, image):  # Call each time AnalysisThread emit a new pix
        self.videoViewer.setInfoText(self.AnalysisThread.getInfoText())

        leftHandKeypoints, leftAccuracy = self.AnalysisThread.getHandData(0)
        rightHandKeypoints, rightAccuracy = self.AnalysisThread.getHandData(1)

        if self.realTimeHandDraw:
            self.handClassifier.leftHandAnalysis.drawHand(
                leftHandKeypoints, leftAccuracy
            )
            self.handClassifier.rightHandAnalysis.drawHand(
                rightHandKeypoints, rightAccuracy
            )

        if self.datasetController.getHandID() == 0:  # Recording left hand
            if type(leftHandKeypoints) != type(None):
                if self.isRecording:
                    if leftAccuracy > self.datasetController.getTresholdValue():
                        self.datasetController.addEntryDataset(
                            leftHandKeypoints, leftAccuracy
                        )
        else:  # Recording right hand
            if type(rightHandKeypoints) != type(None):  # If selected hand detected
                if self.isRecording:
                    if rightAccuracy > self.datasetController.getTresholdValue():
                        self.datasetController.addEntryDataset(
                            rightHandKeypoints, rightAccuracy
                        )

    def changeHandDrawingState(self, state: bool):
        self.realTimeHandDraw = state


app = QtWidgets.QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.show()
sys.exit(app.exec_())
