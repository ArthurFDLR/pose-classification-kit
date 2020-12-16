import os
import numpy as np
from datetime import date
import numpy as np
from pathlib import Path

from .qt import QtWidgets, QtCore, QtGui, pyqtSignal, pyqtSlot
from .Util import SwitchButton, ScrollLabel, mat2QImage, isHandData


class DatasetControllerWidget(QtWidgets.QWidget):
    realTimeHandDraw_Signal = pyqtSignal(bool)
    stylesheet = """
    #Dataset_Controller {
        background-color: white;
        border-radius: 3px;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    }
    QPushButton {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: white;
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
    QComboBox {
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
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.currentFilePath = ""
        self.currentFileHeadLines = ""
        self.poseName = ""
        self.handID = 1
        self.tresholdValue = 0.0
        self.datasetList = []
        self.accuracyList = []
        self.currentDataIndex = 0
        self.datasetSaved = True

        ## Widget style
        self.setObjectName('Dataset_Controller')
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
        self.fileLabel.setMaximumHeight(60)
        self.fileLabel.setMinimumWidth(180)
        self.layout.addWidget(self.fileLabel, 0, 0, 1, 9, QtCore.Qt.AlignTop)

        self.visuCheckbox = QtWidgets.QCheckBox("Visualize imported dataset")
        self.layout.addWidget(self.visuCheckbox, 1, 0)
        self.visuCheckbox.toggled.connect(self.visuCheckboxToggled)
        self.visuCheckbox.setEnabled(False)

        self.minusButton = QtWidgets.QToolButton(cursor=QtCore.Qt.PointingHandCursor, toolTip='Previous sample in dataset')
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
        self.currentIndexLine.setMaximumWidth(25)
        self.currentIndexLine.setEnabled(False)
        self.layout.addWidget(self.currentIndexLine, 1, 2, 1, 1)
        self.currentIndexLine.textChanged.connect(self.userIndexInput)

        self.maxIndexLabel = QtWidgets.QLabel(r"/0")
        self.maxIndexLabel.setEnabled(False)
        self.layout.addWidget(self.maxIndexLabel, 1, 3, 1, 1)

        self.plusButton = QtWidgets.QToolButton(cursor=QtCore.Qt.PointingHandCursor, toolTip='Next sample in dataset')
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

        self.deleteButton = QtWidgets.QPushButton("Delete sample", cursor=QtCore.Qt.PointingHandCursor, toolTip='Remove sample from the dataset')
        self.deleteButton.setEnabled(False)
        self.layout.addWidget(self.deleteButton, 1, 5, 1, 1)
        self.deleteButton.clicked.connect(
            lambda: self.removeEntryDataset(self.currentDataIndex)
        )

        self.layout.addWidget(QtWidgets.QLabel("Recording:"), 1, 7, 1, 1)

        self.recordButton = SwitchButton()
        self.recordButton.setChecked(False)
        self.recordButton.setEnabled(False)
        self.layout.addWidget(self.recordButton, 1, 8, 1, 1)
        self.recordButton.clickedChecked.connect(self.startRecording)

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
                dlg.getFileHeadlines(),
                0,
                dlg.getPoseName(),
                dlg.getHandID(),
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
        self.maxIndexLabel.setText("/" + str(len(self.accuracyList)))
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

    def loadFile(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open dataset", r".\Dataset", "Text Files (*.txt)", options=options
        )
        self.clearDataset()
        currentEntry = []

        if fileName:
            self.clearDataset()
            dataFile = open(fileName)
            fileHeadline = ""
            for i, line in enumerate(dataFile):
                if i == 0:
                    info = line.split(",")
                    fileHeadline += line
                    if len(info) == 3:
                        poseName = info[0]
                        handID = int(info[1])
                        tresholdValue = float(info[2])
                    else:
                        self.fileLabel.setText("Not a supported dataset")
                        break
                else:
                    if line[0] == "#" and line[1] == "#":  # Commentary/headlines
                        fileHeadline += line
                    elif line[0] == "#" and line[1] != "#":  # New entry
                        currentEntry = [[], [], []]
                        accuracy = float(line[1:])
                    elif line[0] == "x":
                        listStr = line[2:].split(" ")
                        for value in listStr:
                            currentEntry[0].append(float(value))
                    elif line[0] == "y":
                        listStr = line[2:].split(" ")
                        for value in listStr:
                            currentEntry[1].append(float(value))
                    elif line[0] == "a":  # Last line of entry
                        listStr = line[2:].split(" ")
                        for value in listStr:
                            currentEntry[2].append(float(value))
                        self.addEntryDataset(currentEntry, accuracy)

            dataFile.close()
            self.updateFileInfo(
                fileName,
                fileHeadline,
                len(self.datasetList),
                poseName,
                handID,
                tresholdValue,
            )
            self.recordButton.setEnabled(True)
            self.visuCheckbox.setChecked(True)
            self.datasetSaved = True
            return True
        return False

    def updateFileInfo(
        self,
        filePath: str = None,
        fileHead: str = None,
        sizeData: int = 0,
        poseName: str = None,
        handID: int = None,
        tresholdValue: int = None,
    ):
        self.visuCheckbox.setEnabled(True)
        if filePath:
            self.currentFilePath = filePath
        if fileHead:
            self.currentFileHeadLines = fileHead
        if poseName:
            self.poseName = poseName
        if handID != None:
            self.handID = handID
        if tresholdValue != None:
            self.tresholdValue = tresholdValue
        self.fileLabel.setText(
            str(self.currentFilePath)
            + "\n  -> {} entries for {} ({} hand) with a minimum accuracy of {}.".format(
                str(sizeData),
                poseName,
                ("right" if handID == 1 else "left"),
                str(tresholdValue),
            )
        )
        self.maxIndexLabel.setText("/" + str(sizeData))
        self.recordButton.setEnabled(True)

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

            if self.handID == 0:
                self.parent.handClassifier.leftHandAnalysis.drawHand(
                    np.array(self.datasetList[self.currentDataIndex]),
                    self.accuracyList[self.currentDataIndex],
                )
            else:
                self.parent.handClassifier.rightHandAnalysis.drawHand(
                    np.array(self.datasetList[self.currentDataIndex]),
                    self.accuracyList[self.currentDataIndex],
                )
        self.currentIndexLine.setText(str(self.currentDataIndex + 1))

    def writeDataToTxt(self):
        """ Save the current dataset to the text file (URL: self.currentFilePath)."""
        if os.path.isfile(self.currentFilePath):
            dataFile = open(self.currentFilePath, "w")  # Open in write 'w' to clear.
            dataFile.write(self.currentFileHeadLines)
            sizeData = len(self.datasetList)
            for entryIndex in range(sizeData):
                dataFile.write("#" + str(self.accuracyList[entryIndex]))
                for i, row in enumerate(self.datasetList[entryIndex]):
                    for j, val in enumerate(row):
                        dataFile.write(
                            "\n{}:".format(["x", "y", "a"][i]) if j == 0 else " "
                        )
                        dataFile.write(str(val))
                dataFile.write("\n\n")
            self.updateFileInfo(sizeData=sizeData)
            self.datasetSaved = True

    def startRecording(self, state: bool):
        self.parent.isRecording = state

    def getTresholdValue(self) -> float:
        return self.tresholdValue

    def getHandID(self) -> int:
        return self.handID

    def getPoseName(self) -> str:
        return self.poseName

    def isSaved(self) -> bool:
        return self.datasetSaved


class CreateDatasetDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(CreateDatasetDialog, self).__init__(parent=parent)

        self.setWindowTitle("Create new dataset")

        self.currentFolder = Path('.').absolute()
        if (self.currentFolder / 'Dataset').is_dir():
            self.currentFolder /= 'Dataset'
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
        
        self.handSelection = HandSelectionWidget(self)
        
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
        self.layout.addWidget(QtWidgets.QLabel("Hand pose name:"), 1, 1, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel("Accuaracy treshold:"), 1, 3, 1, 1)
        self.layout.addWidget(self.tresholdValueLine, 1, 4, 1, 1)
        self.layout.addWidget(self.createButton, 1, 5, 1, 1)

    def createDataset(self):
        self.isRecording = True

        path = self.getSavingFolder()
        folder = self.getPoseName()
        tresholdValue = self.getTresholdValue()
        handID = self.handSelection.getCurrentHandID()

        path /= folder
        if not path.is_dir():  # Create pose directory if missing
            os.mkdir(path)

        path /= ("right_hand" if handID == 1 else "left_hand")
        if path.is_dir():
            self.isRecording = False
            self.createButton.setEnabled(False)
            self.createButton.setText("Dataset allready created")

        else:
            self.createButton.setEnabled(True)
            self.createButton.setText("Create dataset")
            os.mkdir(path)  # Create hand directory if missing

            path /= 'data.txt'
            currentFile = open(path, "w+")
            currentFile.write(self.getFileHeadlines())
            currentFile.close()
            self.accept()
            self.currentFilePath = path

    def getFileHeadlines(self):
        folder = self.getPoseName()
        tresholdValue = self.getTresholdValue()
        handID = self.handSelection.getCurrentHandID()

        output = ""
        output += folder + "," + str(handID) + "," + str(tresholdValue) + "\n"
        output += "## Data generated the " + str(date.today()) + " labelled " + folder
        output += " (" + ("right hand" if handID == 1 else "left hand")
        output += ") with a global accuracy higher than " + str(tresholdValue)
        output += ", based on OpenPose estimation.\n"

        output += "## Data format: Coordinates x, y and accuracy of estimation a\n\n"

        return output

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

    def getHandID(self) -> int:
        return self.handSelection.getCurrentHandID()

    def getFilePath(self):
        return self.currentFilePath

    def resizeEvent(self, event):
        self.folderButton.setFixedHeight(self.folderLabel.height())


class HandSelectionWidget(QtWidgets.QWidget):
    changeHandSelection = pyqtSignal(int)

    def __init__(self, parent=None):
        super(HandSelectionWidget, self).__init__(parent)
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.parent = parent

        self.layout.addWidget(QtWidgets.QLabel("Hand focus:"), 0, 0)
        self.rightCheckbox = QtWidgets.QCheckBox("Right")
        self.leftCheckbox = QtWidgets.QCheckBox("Left")
        self.layout.addWidget(self.leftCheckbox, 0, 1)
        self.layout.addWidget(self.rightCheckbox, 0, 2)

        horSpacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.layout.addItem(horSpacer, 0, 3)

        self.rightCheckbox.toggled.connect(
            lambda check: self.leftCheckbox.setChecked(not check)
        )
        self.leftCheckbox.toggled.connect(
            lambda check: self.rightCheckbox.setChecked(not check)
        )
        self.rightCheckbox.toggled.connect(
            lambda check: self.changeHandSelection.emit(1 if check else 0)
        )

        self.rightCheckbox.setChecked(True)

    def getCurrentHandID(self):
        return 1 if self.rightCheckbox.isChecked() else 0
