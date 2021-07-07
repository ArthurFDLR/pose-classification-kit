import os
import numpy as np
from datetime import date
import numpy as np
from pathlib import Path
import json

from .imports.qt import QtWidgets, QtCore, QtGui, pyqtSignal, pyqtSlot
from .imports.openpose import OPENPOSE_LOADED

if OPENPOSE_LOADED:
    from .imports.openpose import op
from ..config import DATASETS_PATH
from ..datasets.body_models import BODY25


class ScrollLabel(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()

        self.setWidgetResizable(True)
        content = QtWidgets.QWidget(self)
        self.setWidget(content)
        lay = QtWidgets.QVBoxLayout(content)
        self.label = QtWidgets.QLabel(content)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lay.addWidget(self.label)

    def setText(self, text):
        self.label.setText(text)


class DatasetControllerWidget(QtWidgets.QWidget):
    realTimeHandDraw_Signal = pyqtSignal(bool)
    stylesheet = """
    #Dataset_Controller {
        background-color: white;
        border-radius: 3px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    #Dataset_Controller:disabled {
        background-color: #e8e8e8;
    }

    QPushButton {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: white;
    }
    QPushButton:hover {
        border-color: rgb(139, 173, 228);
    }
    QPushButton:pressed {
        color: #cbcbcb;
    }
    QPushButton:disabled {
        background: #e8e8e8;
    }

    QToolButton {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: white;
    }
    QToolButton:hover {
        border-color: rgb(139, 173, 228);
    }
    QToolButton:disabled {
        background: #e8e8e8;
    }

    #Record_Button {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: #ffb3b3;
    }
    #Record_Button:checked {
        background: #b3ffb3;
    }
    #Record_Button:disabled {
        background: #e8e8e8;
    }
    #Record_Button:hover {
        border-color: rgb(139, 173, 228);
    }

    QComboBox {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: white;
    }
    QComboBox:disabled {
        background: #e8e8e8;
    }

    QLabel {
        font-size: 16px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }

    QLineEdit {
        font-size: 16px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }

    QCheckBox {
        font-size: 16px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    QCheckBox:disabled {
        background: #e8e8e8;
    }
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.currentFilePath = ""
        self.currentFileInfos = ""
        self.poseName = ""
        self.focusID = 1
        self.sizeData = 0
        self.tresholdValue = 0.0
        self.datasetList = []
        self.accuracyList = []
        self.currentDataIndex = 0
        self.datasetSaved = True

        ## Widget style
        self.setObjectName("Dataset_Controller")
        self.setEnabled(False)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(self.stylesheet)

        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(10)
        effect.setOffset(0, 0)
        effect.setColor(QtCore.Qt.gray)
        self.setGraphicsEffect(effect)

        ## Widgets initialisation
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        self.fileLabel = ScrollLabel()
        self.fileLabel.setText("No file selected")
        self.fileLabel.setMinimumHeight(90)
        self.fileLabel.setMaximumHeight(90)
        self.fileLabel.setMinimumWidth(180)
        self.layout.addWidget(self.fileLabel, 0, 0, 1, 9, QtCore.Qt.AlignTop)

        self.visuCheckbox = QtWidgets.QCheckBox("Visualize dataset")
        self.layout.addWidget(self.visuCheckbox, 1, 0)
        self.visuCheckbox.toggled.connect(self.visuCheckboxToggled)
        self.visuCheckbox.setEnabled(False)

        self.minusButton = QtWidgets.QToolButton(
            cursor=QtCore.Qt.PointingHandCursor, toolTip="Previous sample in dataset"
        )
        self.minusButton.setArrowType(QtCore.Qt.LeftArrow)
        self.layout.addWidget(self.minusButton, 1, 1, 1, 1)
        self.minusButton.setEnabled(False)
        self.minusButton.clicked.connect(
            lambda: self.setCurrentDataIndex(self.currentDataIndex - 1)
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("left"),
            self,
            lambda: self.setCurrentDataIndex(self.currentDataIndex - 1),
        )

        self.currentIndexLine = QtWidgets.QLineEdit(str(self.currentDataIndex))
        self.currentIndexLine.setValidator(QtGui.QDoubleValidator())
        self.currentIndexLine.setMaximumWidth(40)
        self.currentIndexLine.setEnabled(False)
        self.layout.addWidget(self.currentIndexLine, 1, 2, 1, 1)
        self.currentIndexLine.textChanged.connect(self.userIndexInput)

        self.maxIndexLabel = QtWidgets.QLabel(r"/0")
        self.maxIndexLabel.setEnabled(False)
        self.layout.addWidget(self.maxIndexLabel, 1, 3, 1, 1)

        self.plusButton = QtWidgets.QToolButton(
            cursor=QtCore.Qt.PointingHandCursor, toolTip="Next sample in dataset"
        )
        self.plusButton.setArrowType(QtCore.Qt.RightArrow)
        self.layout.addWidget(self.plusButton, 1, 4, 1, 1)
        self.plusButton.setEnabled(False)
        self.plusButton.clicked.connect(
            lambda: self.setCurrentDataIndex(self.currentDataIndex + 1)
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("right"),
            self,
            lambda: self.setCurrentDataIndex(self.currentDataIndex + 1),
        )

        self.deleteButton = QtWidgets.QPushButton(
            "Delete sample",
            cursor=QtCore.Qt.PointingHandCursor,
            toolTip="Remove sample from the dataset",
        )
        self.deleteButton.setEnabled(False)
        self.layout.addWidget(self.deleteButton, 1, 5, 1, 1)
        self.deleteButton.clicked.connect(
            lambda: self.removeEntryDataset(self.currentDataIndex)
        )

        self.recordButton = QtWidgets.QPushButton(
            "Record samples",
            cursor=QtCore.Qt.PointingHandCursor,
            toolTip="Start and stop sample recording",
        )
        self.recordButton.setObjectName("Record_Button")
        self.recordButton.setCheckable(True)
        self.recordButton.setChecked(False)
        self.recordButton.setEnabled(False)
        self.recordButton.clicked.connect(self.startRecording)
        self.layout.addWidget(self.recordButton, 1, 7, 1, 1)

        horSpacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.layout.addItem(horSpacer, 1, 6)

        verSpacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.layout.addItem(verSpacer, 2, 0)

    def createDataset(self):
        dlg = CreateDatasetDialog(self)
        if dlg.exec_():
            self.clearDataset()
            self.updateFileInfo(
                dlg.getFilePath(),
                dlg.getFileInfos(),
                0,
                dlg.getPoseName(),
                dlg.getFocusID(),
                dlg.getTresholdValue(),
            )
            self.setCurrentDataIndex(0)

    def addEntryDataset(self, keypoints, accuracy: float):
        """Add keypoints and accuracy of a hand pose to the local dataset.

        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        """
        self.datasetList.append(keypoints)
        self.accuracyList.append(accuracy)
        self.maxIndexLabel.setText("/" + str(len(self.datasetList)))
        self.datasetSaved = False

    def removeEntryDataset(self, index: int):
        """Remove keypoints and accuracy referenced by its index from the local dataset.

        Args:
            index (int): Index in list of the entry removed.
        """
        self.datasetList = self.datasetList[:index] + self.datasetList[index + 1 :]
        self.accuracyList = self.accuracyList[:index] + self.accuracyList[index + 1 :]
        maxIndex = len(self.accuracyList)
        self.maxIndexLabel.setText("/" + str(maxIndex))
        index = min(index, maxIndex - 1)
        self.setCurrentDataIndex(index)
        self.datasetSaved = False

    def clearDataset(self):
        self.datasetList = []
        self.accuracyList = []
        self.datasetSaved = True

    def userIndexInput(self, indexStr: str):
        if indexStr.isdigit():
            self.setCurrentDataIndex(int(indexStr) - 1)
        elif len(indexStr) == 0:
            pass
        else:
            self.currentIndexLine.setText(str(self.currentDataIndex + 1))

    def visuCheckboxToggled(self, state: bool):
        self.realTimeHandDraw_Signal.emit(not state)
        self.plusButton.setEnabled(state)
        self.minusButton.setEnabled(state)
        self.currentIndexLine.setEnabled(state)
        self.maxIndexLabel.setEnabled(state)
        self.deleteButton.setEnabled(state)
        self.setCurrentDataIndex(0)

    def loadFileJSON(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            str(DATASETS_PATH),
            "Text Files (*.json)",
            options=options,
        )

        if fileName:
            self.clearDataset()

            with open(fileName) as f:
                data = json.load(f)

            for entry in data["data"]:
                self.addEntryDataset(
                    np.array([entry["x"], entry["y"], entry["a"]]),
                    float(entry["detection_accuracy"]),
                )

            self.updateFileInfo(
                filePath=fileName,
                fileInfo={"info": data["info"], "data": []},
                sizeData=data["info"]["nbr_entries"],
                poseName=data["info"]["label"],
                focusID=data["info"]["focus_id"],
                tresholdValue=data["info"]["threshold_value"],
            )

            self.recordButton.setEnabled(True)
            self.setEnabled(True)
            self.visuCheckbox.setChecked(True)
            self.datasetSaved = True
            return True
        return False

    def updateFileInfo(
        self,
        filePath: str = None,
        fileInfo: str = None,
        sizeData: int = None,
        poseName: str = None,
        focusID: int = None,
        tresholdValue: int = None,
    ):
        self.visuCheckbox.setEnabled(True)
        if filePath:
            self.currentFilePath = filePath
        if fileInfo:
            self.currentFileInfos = fileInfo
        if sizeData:
            self.sizeData = sizeData
            self.maxIndexLabel.setText("/" + str(self.sizeData))
        if poseName:
            self.poseName = poseName
        if focusID != None:
            self.focusID = focusID
        if tresholdValue != None:
            self.tresholdValue = tresholdValue
        self.fileLabel.setText(
            str(self.currentFilePath)
            + "\n  -> {} entries for {} ({} hand) with a minimum accuracy of {}.".format(
                str(self.sizeData),
                self.poseName,
                ["left_hand", "right_hand", "body"][self.focusID],
                str(self.tresholdValue),
            )
        )
        # self.maxIndexLabel.setText("/" + str(self.sizeData))
        self.recordButton.setEnabled(True)
        self.setEnabled(True)

    def setCurrentDataIndex(self, index: int):
        if len(self.datasetList) == 0:
            self.currentDataIndex = 0
            self.parent.handClassifier.leftHandAnalysis.drawHand(None, 0.0)
            self.parent.handClassifier.rightHandAnalysis.drawHand(None, 0.0)
        else:
            if index >= len(self.datasetList):
                index = 0
            if index < 0:
                index = len(self.datasetList) - 1
            self.currentDataIndex = index

            if self.focusID == 0:
                self.parent.handClassifier.leftHandAnalysis.drawHand(
                    np.array(self.datasetList[self.currentDataIndex]),
                    self.accuracyList[self.currentDataIndex],
                )
            elif self.focusID == 1:
                self.parent.handClassifier.rightHandAnalysis.drawHand(
                    np.array(self.datasetList[self.currentDataIndex]),
                    self.accuracyList[self.currentDataIndex],
                )
            elif self.focusID == 2:
                self.parent.bodyClassifier.bodyAnalysis.drawBody(
                    np.array(self.datasetList[self.currentDataIndex]),
                    self.accuracyList[self.currentDataIndex],
                )
        self.currentIndexLine.setText(str(self.currentDataIndex + 1))

    def writeDataToJSON(self):
        """ Save the current dataset to the JSON file (URL: self.currentFilePath)."""
        if os.path.isfile(self.currentFilePath):
            fileData = self.currentFileInfos
            fileData["info"]["nbr_entries"] = len(self.datasetList)
            fileData["data"] = []
            self.updateFileInfo(sizeData=len(self.datasetList))
            print(len(self.datasetList))
            for accuracy, data in zip(self.accuracyList, self.datasetList):
                fileData["data"].append(
                    {
                        "detection_accuracy": float(accuracy),
                        "x": data[0].tolist(),
                        "y": data[1].tolist(),
                        "a": data[2].tolist(),
                    }
                )

            with open(self.currentFilePath, "w") as outfile:
                json.dump(fileData, outfile, indent=4)

            self.datasetSaved = True

    def startRecording(self, state: bool):
        self.parent.isRecording = state

    def getTresholdValue(self) -> float:
        return self.tresholdValue

    def getFocusID(self) -> int:
        return self.focusID

    def getPoseName(self) -> str:
        return self.poseName

    def isSaved(self) -> bool:
        return self.datasetSaved


class CreateDatasetDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(CreateDatasetDialog, self).__init__(parent=parent)

        self.setWindowTitle("Create new dataset")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)

        self.currentFolder = DATASETS_PATH
        self.currentFilePath = None
        self.currentPoseName = "Default"
        self.currentTresholdValue = 0.0

        ## Widgets initialisation
        self.folderLabel = ScrollLabel()
        self.folderLabel.setText(str(self.currentFolder))
        self.folderLabel.setMaximumHeight(35)
        self.folderLabel.setMinimumWidth(200)

        self.folderButton = QtWidgets.QPushButton("Change root folder")
        self.folderButton.clicked.connect(self.changeSavingFolder)

        self.handSelection = FocusSelectionWidget(self)

        self.poseNameLine = QtWidgets.QLineEdit(self.currentPoseName)
        self.poseNameLine.textChanged.connect(self.changePoseName)

        self.tresholdValueLine = QtWidgets.QLineEdit(str(self.currentTresholdValue))
        onlyDouble = QtGui.QDoubleValidator()
        self.tresholdValueLine.setValidator(onlyDouble)
        self.tresholdValueLine.textChanged.connect(self.changeTresholdValue)

        self.createButton = QtWidgets.QPushButton("Create dataset")
        self.createButton.clicked.connect(self.createDataset)

        ## Structure
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.folderLabel, 0, 0, 1, 5, QtCore.Qt.AlignTop)
        self.layout.addWidget(self.folderButton, 0, 5, 1, 1, QtCore.Qt.AlignTop)
        self.layout.addWidget(self.handSelection, 1, 0, 1, 1)
        self.layout.addWidget(self.poseNameLine, 1, 2, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel("Label:"), 1, 1, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel("Accuracy threshold:"), 1, 3, 1, 1)
        self.layout.addWidget(self.tresholdValueLine, 1, 4, 1, 1)
        self.layout.addWidget(self.createButton, 1, 5, 1, 1)
        self.layout.setRowStretch(0, 0)
        self.layout.setRowStretch(1, 0)
        self.layout.setRowStretch(2, 1)

    def createDataset(self):
        self.isRecording = True
        path = self.getSavingFolder()
        focusID = self.handSelection.getCurrentFocusID()
        fileName = (
            self.getPoseName()
            + "_"
            + ["left_hand", "right_hand", "body"][focusID]
            + ".json"
        )
        path /= fileName
        if path.is_file():
            self.isRecording = False
            self.createButton.setEnabled(False)
            self.createButton.setText("Dataset allready created")

        else:
            self.createButton.setEnabled(True)
            self.createButton.setText("Create dataset")
            with open(path, "w+") as outfile:
                json.dump(self.getFileInfos(), outfile, indent=4, ensure_ascii=False)
            self.accept()
            self.currentFilePath = path

    def getFileHeadlines(self):
        folder = self.getPoseName()
        tresholdValue = self.getTresholdValue()
        handID = self.handSelection.getCurrentFocusID()

        output = ""
        output += folder + "," + str(handID) + "," + str(tresholdValue) + "\n"
        output += "## Data generated the " + str(date.today()) + " labelled " + folder
        output += " (" + ("right hand" if handID == 1 else "left hand")
        output += ") with a global accuracy higher than " + str(tresholdValue)
        output += ", based on OpenPose estimation.\n"

        output += "## Data format: Coordinates x, y and accuracy of estimation a\n\n"

        return output

    def getFileInfos(self):
        info = {
            "info": {
                "label": self.getPoseName(),
                "focus": ["left_hand", "right_hand", "body"][
                    self.handSelection.getCurrentFocusID()
                ],
                "nbr_entries": 0,
                "threshold_value": self.getTresholdValue(),
                "focus_id": self.handSelection.getCurrentFocusID(),
            },
            "data": [],
        }
        if self.handSelection.getCurrentFocusID() == 2 and OPENPOSE_LOADED:
            info["info"]["BODY25_Mapping"] = BODY25.mapping
            info["info"]["BODY25_Pairs"] = BODY25.pairs
        return info

    @pyqtSlot()
    def changeSavingFolder(self):
        path_str = str(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        )
        if len(path_str) > 0:
            self.folderLabel.setText(path_str)
            self.currentFolder = Path(path_str)

    @pyqtSlot(str)
    def changePoseName(self, name: str):
        if not self.createButton.isEnabled():
            self.createButton.setEnabled(True)
            self.createButton.setText("Create dataset")
        self.currentPoseName = name

    @pyqtSlot(str)
    def changeTresholdValue(self, value: str):
        try:
            self.currentTresholdValue = float(value.replace(",", "."))
        except:
            self.currentTresholdValue = 0.0

    def getSavingFolder(self):
        return self.currentFolder

    def getPoseName(self) -> str:
        return self.currentPoseName

    def getTresholdValue(self) -> float:
        return self.currentTresholdValue

    def getFocusID(self) -> int:
        return self.handSelection.getCurrentFocusID()

    def getFilePath(self):
        return self.currentFilePath

    def resizeEvent(self, event):
        self.folderButton.setFixedHeight(self.folderLabel.height())


class FocusSelectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(FocusSelectionWidget, self).__init__(parent)
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.parent = parent

        self.layout.addWidget(QtWidgets.QLabel("Focus:"), 0, 0)
        self.rightCheckbox = QtWidgets.QCheckBox("Right hand")
        self.leftCheckbox = QtWidgets.QCheckBox("Left hand")
        self.bodyCheckbox = QtWidgets.QCheckBox("Body")
        self.layout.addWidget(self.leftCheckbox, 0, 1)
        self.layout.addWidget(self.rightCheckbox, 0, 2)
        self.layout.addWidget(self.bodyCheckbox, 0, 3)

        horSpacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.layout.addItem(horSpacer, 0, 3)

        group = QtWidgets.QButtonGroup(self)
        group.addButton(self.rightCheckbox)
        group.addButton(self.leftCheckbox)
        group.addButton(self.bodyCheckbox)
        group.buttonClicked.connect(self.toggleFocus)

        self.bodyCheckbox.setChecked(True)
        self.focusID = 2

    def toggleFocus(self, btn):
        label = btn.text()
        if label == "Left hand":
            self.focusID = 0
        elif label == "Right hand":
            self.focusID = 1
        elif label == "Body":
            self.focusID = 2

    def getCurrentFocusID(self):
        return self.focusID
