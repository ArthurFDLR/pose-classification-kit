import queue
import time
from datetime import date
import cv2
import sys
import os
import pyqtgraph as pg
import numpy as np
import pathlib

try:
    from Util import SwitchButton, ScrollLabel, VLine, mat2QImage, isHandData
except:
    from .Util import SwitchButton, ScrollLabel, VLine, mat2QImage, isHandData


from PyQt5 import QtWidgets as Qtw
from PyQt5.QtCore import Qt, QThread,  pyqtSignal, pyqtSlot, QSize, QBuffer
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator, QColor, QIcon, QKeySequence, QMessageBox
from PyQt5.QtMultimedia import QCameraInfo, QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

# Path to OpenPose installation folder on your system.

openposePATH = pathlib.Path('C:/') / 'Program files' / 'OpenPose'
try:
    sys.path.append(str(openposePATH / 'build' / 'python' / 'openpose' / 'Release'))
    releasePATH = openposePATH / 'build' / 'x64' / 'Release'
    binPATH = openposePATH / 'build' / 'bin'
    modelsPATH = openposePATH / 'models'
    os.environ['PATH'] = os.environ['PATH'] + ';' + str(releasePATH) + ';' + str(binPATH) + ';'
    import pyopenpose as op
    OPENPOSE_LOADED = True
except:
    OPENPOSE_LOADED = False
    print('OpenPose ({}) loading failed.'.format(str(openposePATH)))

SHOW_TF_WARNINGS = False
if not SHOW_TF_WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Avoid the annoying tf warnings

try:
    import tensorflow as tf
    
    GPU_LIST = tf.config.experimental.list_physical_devices('GPU')
    if GPU_LIST:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in GPU_LIST:
                tf.config.experimental.set_memory_growth(gpu, True) #Prevent Tensorflow to take all GPU memory
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(GPU_LIST), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    TF_LOADED = True
except:
    TF_LOADED = False


class CameraInput(Qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(CameraInput, self).__init__(*args, **kwargs)

        self.available_cameras = QCameraInfo.availableCameras()
        
        if not self.available_cameras:
            print('No camera')
            pass #quit

        self.buffer = QBuffer
        #self.lastImage = QImage('.\\Data\\tempInit.png')
        self.lastImage = QPixmap(10, 10).toImage()
        self.lastID = None
        self.save_path = ""
        self.tmpUrl = str(pathlib.Path(__file__).parent.absolute() / 'Data' / 'tmp.png')

        self.capture = None

        self.select_camera(0)
    
    def refreshCameraList(self):
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            print('No camera')
            return None
        self.camera.stop()
        self.select_camera(0)
        return self.available_cameras

    def getAvailableCam(self):
        return self.available_cameras

    def select_camera(self, i):
        if len(self.available_cameras) > 0:
            self.camera = QCamera(self.available_cameras[i])
            self.camera.setCaptureMode(QCamera.CaptureStillImage)
            self.camera.start()

            self.capture = QCameraImageCapture(self.camera)
            self.capture.setCaptureDestination(QCameraImageCapture.CaptureToBuffer)

            self.capture.imageCaptured.connect(self.storeLastFrame)

            self.current_camera_name = self.available_cameras[i].description()
            self.save_seq = 0
        else:
            print('No camera.')

    def getLastFrame(self):
        if self.capture:
            imageID = self.capture.capture()
            return self.qImageToMat(self.lastImage)
        else: 
            return None

    def storeLastFrame(self, idImg:int, preview:QImage):
        self.lastImage = preview
        self.lastID = idImg
    
    def qImageToMat(self, incomingImage):
        incomingImage.save(self.tmpUrl, 'png')
        mat = cv2.imread(self.tmpUrl)
        return mat
    
    def deleteTmpImage(self):
        os.remove(self.tmpUrl)
        self.tmpUrl = None


''' No temporary files but too slow
    def qImageToMat(self, incomingImage):
        incomingImage = incomingImage.convertToFormat(4) #Set to format RGB32
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits() #Get pointer to first pixel
        ptr.setsize(height * width * 4) #Get pointer to full image
        arr = np.array(ptr).reshape(height, width, 4)  #Copies the data
        arr = np.delete(arr, 3, 2) #Delete alpha channel
        return arr
'''


class VideoAnalysisThread(QThread):
    newPixmap = pyqtSignal(QImage)
    newMat = pyqtSignal(np.ndarray)
    def __init__(self, videoSource, qimageEmission:bool=True):
        super().__init__()
        self.infoText = ''
        self.personID = 0
        self.running = False
        self.videoSource = videoSource
        self.qimageEmission = qimageEmission

        ## Starting OpenPose ##
        #######################
        if OPENPOSE_LOADED:
            params = dict()
            params["model_folder"] = str(modelsPATH)
            params["face"] = False
            params["hand"] = True
            #params["body"] = 0
            #params["hand_detector"] = 2
            params["disable_multi_thread"] = False
            netRes = 15 #Default 22
            params["net_resolution"] = "-1x"+str(16*netRes) 

            self.opWrapper = op.WrapperPython()
            self.datum = op.Datum()
            self.opWrapper.configure(params)
            self.opWrapper.start()

        
        self.lastTime = time.time()
        self.emissionFPS = 3.0
        self.fixedFps = True

        self.videoWidth = 1280 
        self.videoHeight = 720
    
    def run(self):
        while OPENPOSE_LOADED:
            if self.running:
                if (time.time() - self.lastTime > 1.0/self.emissionFPS) or not self.fixedFps:
                    self.lastTime = time.time()

                    frame = self.videoSource.getLastFrame()
                    if type(frame) != type(None): #Check if frame exist, frame!=None is ambigious when frame is an array
                        frame = self.resizeCvFrame(frame, 0.5)
                        self.datum.cvInputData = frame
                        self.opWrapper.emplaceAndPop([self.datum])
                        frameOutput = self.datum.cvOutputData
                        self.newMat.emit(frameOutput)

                        if self.qimageEmission:
                            image = mat2QImage(frameOutput)
                            self.newPixmap.emit(image.scaled(self.videoWidth, self.videoHeight, Qt.KeepAspectRatio))

    @pyqtSlot(bool)
    def setState(self, s:bool):
        self.running = s
    
    def setResolutionStream(self, width:int, height:int):
        self.videoHeight = height
        self.videoWidth = width
    
    def setEmissionSpeed(self, fixedFPS:bool, fps:int):
        self.fixedFps=fixedFPS
        if self.fixedFps:
            self.emissionFPS = fps
    
    def getHandData(self, handID:int):
        ''' Return the key points of the hand seen in the image (cf. videoSource).
        
        Args:
            handID (int): 0 -> Left hand | 1 -> Right hand
        
        Returns:
            np.ndarray((3,21),float): Coordinates x, y and the accuracy score for each 21 key points.
                                      None if the given hand is not detected.
        '''
        outputArray = None

        handKeypoints = np.array(self.datum.handKeypoints)
        nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim >2 else 0
        handAccuaracyScore = .0
        if nbrPersonDetected > 0:
            handAccuaracyScore = handKeypoints[handID, self.personID].T[2].sum()
            handDetected = (handAccuaracyScore > 1.0)
            if handDetected:
                handKeypoints = handKeypoints[handID, self.personID]

                lengthFingers = [np.sqrt((handKeypoints[0,0] - handKeypoints[i,0])**2 + (handKeypoints[0,1] - handKeypoints[i,1])**2) for i in [1,5,9,13,17]] #Initialize with the length of the first segment of each fingers
                for i in range(3): #Add length of other segments of each fingers
                    for j in range(len(lengthFingers)):
                        lengthFingers[j] += np.sqrt((handKeypoints[1+j*4+i+1, 0] - handKeypoints[1+j*4+i, 0])**2 + (handKeypoints[1+j*4+i+1, 1] - handKeypoints[1+j*4+i, 1])**2)
                normMax = max(lengthFingers)

                handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
                handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]
                outputArray = np.array([(handKeypoints.T[0] - handCenterX)/normMax,
                                        -(handKeypoints.T[1] - handCenterY)/normMax,
                                        (handKeypoints.T[2])])
        return outputArray, handAccuaracyScore
    
    def getBodyData(self):
        if len(self.datum.poseKeypoints.shape) > 0:
            poseKeypoints = self.datum.poseKeypoints[self.personID]
            return poseKeypoints
        else:
            return None
    
    def getInfoText(self) -> str:
        handKeypoints = np.array(self.datum.handKeypoints)
        nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim >2 else 0

        self.infoText = ''
        self.infoText += str(nbrPersonDetected) + (' person detected' if nbrPersonDetected<2 else  ' person detected')

        if nbrPersonDetected > 0:
            leftHandDetected = (handKeypoints[0, self.personID].T[2].sum() > 1.0)
            rightHandDetected = (handKeypoints[1, self.personID].T[2].sum() > 1.0)
            if rightHandDetected and leftHandDetected:
                self.infoText += ', both hands of person ' + str(self.personID+1) + ' detected.'
            elif rightHandDetected or leftHandDetected:
                self.infoText += ', ' + ('Right' if rightHandDetected else 'Left') + ' hand of person ' + str(self.personID+1) + ' detected.'
            else:
                self.infoText += ', no hand of person ' + str(self.personID+1) + ' detected.'

        return self.infoText
    
    def getFingerLength(self, fingerData):
        length = .0
        for i in range(fingerData.shape[0]-1):
            length += np.sqrt((fingerData[i+1,0] - fingerData[i,0])**2 + (fingerData[i+1,1] - fingerData[i,1])**2)
        return length
    
    def resizeCvFrame(self, frame, ratio:float):
        width = int(frame.shape[1] * ratio) 
        height = int(frame.shape[0] * ratio) 
        dim = (width, height) 
        # resize image in down scale
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
        return resized

    def isRaisingHand(self):
        poseKeypoints = self.getBodyData()
        raisingRight = False
        raisingLeft = False
        if type(poseKeypoints) != type(None):
            rightHand_x, rightHand_y, rightHand_a = poseKeypoints[4]
            leftHand_x, leftHand_y, leftHand_a = poseKeypoints[7]
            rightShoulder_x, rightShoulder_y, rightShoulder_a = poseKeypoints[2]
            leftShoulder_x, leftShoulder_y, leftShoulder_a = poseKeypoints[5]
            
            shoulderSlope = (rightShoulder_y - leftShoulder_y) / (rightShoulder_x - leftShoulder_x)
            shoulderOri = rightShoulder_y - shoulderSlope * rightShoulder_x

            if leftHand_a > 0.1:
                raisingLeft = leftHand_y < (shoulderSlope * leftHand_x + shoulderOri) # y axis oriented from top to down in images
                raisingLeft = raisingLeft and leftHand_y < poseKeypoints[6,1] # Check if hand above elbow
            else:
                raisingLeft = False
            if rightHand_a > 0.1:
                raisingRight = rightHand_y < (shoulderSlope * rightHand_x + shoulderOri)
                raisingRight = raisingRight and rightHand_y < poseKeypoints[3,1]
            else:
                raisingRight = False
        
        return raisingLeft, raisingRight


class VideoViewer(Qtw.QGroupBox):
    changeCameraID_signal = pyqtSignal
    def __init__(self, availableCameras):
        super().__init__('Camera feed')

        self.availableCameras = availableCameras

        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)

        self.rawCamFeed = Qtw.QLabel(self)
        
        self.infoLabel = Qtw.QLabel('No info')

        self.refreshButton = Qtw.QPushButton('Refresh camera list')
    
        self.camera_selector = Qtw.QComboBox()
        self.camera_selector.addItems([c.description() for c in self.availableCameras])
        

        if OPENPOSE_LOADED:
            self.layout.addWidget(self.rawCamFeed,0,0,1,3)
            self.layout.addWidget(self.infoLabel,1,2,1,1)
            self.layout.addWidget(self.refreshButton, 1,0,1,1)
            self.layout.addWidget(self.camera_selector,1,1,1,1)
        else:
            self.layout.addWidget(Qtw.QLabel('Video analysis impossible.\nCheck OpenPose installation.'),0,0,1,1)

        self.autoAdjustable = False

    @pyqtSlot(QImage)
    def setImage(self, image:QImage):
        self.currentPixmap = QPixmap.fromImage(image)
        self.rawCamFeed.setPixmap(self.currentPixmap.scaled(self.rawCamFeed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def resizeEvent(self, event):
        if self.autoAdjustable:
            try:
                w = self.rawCamFeed.width()
                h = self.rawCamFeed.height()
                self.rawCamFeed.setPixmap(self.currentPixmap.scaled(w,h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.rawCamFeed.setMinimumSize(100,100)

                w = self.pepperCamFeed.width()
                h = self.pepperCamFeed.height()
                self.pepperCamFeed.setPixmap(self.pepperCamFeed.scaled(w,h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.pepperCamFeed.setMinimumSize(100,100)
            except:
                pass
    
    def setVideoSize(self, width:int, height:int):
        self.rawCamFeed.setFixedSize(width,height)
    
    def setInfoText(self, info:str):
        if info:
            self.infoLabel.setText(info)
        else:
            self.infoLabel.setText('')


class CreateDatasetDialog(Qtw.QDialog):
    def __init__(self, parent = None):
        super(CreateDatasetDialog, self).__init__(parent = parent)
        
        self.setWindowTitle("Create new dataset")

        self.currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.currentFolder += r'\Datasets'
        self.currentFilePath = None
        self.currentPoseName = 'Default'
        self.currentTresholdValue = .0
        ## Widgets initialisation
        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)

        self.folderLabel = ScrollLabel()
        self.folderLabel.setText(self.currentFolder)
        self.folderLabel.setMaximumHeight(35)
        self.folderLabel.setMinimumWidth(200)
        #self.folderLabel.setStyleSheet("background-color:#000000;")
        self.layout.addWidget(self.folderLabel, 0,0,1,5, Qt.AlignTop)

        self.folderButton = Qtw.QPushButton('Change root folder')
        self.folderButton.clicked.connect(self.changeSavingFolder)
        self.layout.addWidget(self.folderButton, 0,5,1,1, Qt.AlignTop)

        self.handSelection = HandSelectionWidget(self)
        self.layout.addWidget(self.handSelection, 1,0,1,1)
        #self.handSelection.changeHandSelection.connect(parent.changeHandID)

        self.layout.addWidget(Qtw.QLabel('Hand pose name:'), 1,1,1,1)
        self.poseNameLine = Qtw.QLineEdit(self.currentPoseName)
        self.layout.addWidget(self.poseNameLine, 1,2,1,1)
        self.poseNameLine.textChanged.connect(self.changePoseName)

        self.layout.addWidget(Qtw.QLabel('Accuaracy treshold:'), 1,3,1,1)
        self.tresholdValueLine = Qtw.QLineEdit(str(self.currentTresholdValue))
        onlyDouble = QDoubleValidator()
        self.tresholdValueLine.setValidator(onlyDouble)
        self.layout.addWidget(self.tresholdValueLine, 1,4,1,1)
        self.tresholdValueLine.textChanged.connect(self.changeTresholdValue)

        self.createButton = Qtw.QPushButton('Create dataset')
        self.layout.addWidget(self.createButton,1,5,1,1)
        self.createButton.clicked.connect(self.createDataset)
        #verticalSpacer = Qtw.QSpacerItem(0, 0, Qtw.QSizePolicy.Minimum, Qtw.QSizePolicy.Expanding)
        #self.layout.addItem(verticalSpacer, 2, 0, Qt.AlignTop)
    
    def createDataset(self):
        self.isRecording = True

        path = self.getSavingFolder()
        folder = self.getPoseName()
        tresholdValue = self.getTresholdValue()
        handID = self.handSelection.getCurrentHandID()

        path += '\\' + folder
        if not os.path.isdir(path): #Create pose directory if missing
            os.mkdir(path)

        path += '\\' + ('right_hand' if handID == 1 else 'left_hand')
        if os.path.isdir(path):
            self.isRecording = False
            self.createButton.setEnabled(False)
            self.createButton.setText('Dataset allready created')

        else:
            self.createButton.setEnabled(True)
            self.createButton.setText('Create dataset')
            os.mkdir(path) #Create hand directory if missing

            path += r'\data.txt'
            currentFile = open(path,"w+")
            currentFile.write(self.getFileHeadlines())
            currentFile.close()
            self.accept()
            self.currentFilePath = path
    
    def getFileHeadlines(self):
        path = self.getSavingFolder()
        folder = self.getPoseName()
        tresholdValue = self.getTresholdValue()
        handID = self.handSelection.getCurrentHandID()
        output = ''
        output += folder + ',' + str(handID) + ',' + str(tresholdValue) + '\n'
        output += '## Data generated the ' + str(date.today()) + ' labelled ' + folder
        output +=  ' (' + ('right hand' if handID == 1 else 'left hand') + ') with a global accuracy higher than ' + str(tresholdValue) + ', based on OpenPose estimation.\n'
        output += '## Data format: Coordinates x, y and accuracy of estimation a\n\n'
        return output
    
    @pyqtSlot()
    def changeSavingFolder(self):
        self.currentFolder = str(Qtw.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.folderLabel.setText(self.currentFolder)
    
    @pyqtSlot(str)
    def changePoseName(self, name:str):
        self.currentPoseName = name
    
    @pyqtSlot(str)
    def changeTresholdValue(self, value:str):
        try:
            self.currentTresholdValue = float(value.replace(',','.'))
        except:
            self.currentTresholdValue = .0

    def getSavingFolder(self)-> str:
        return self.currentFolder

    def getPoseName(self)->str:
        return self.currentPoseName
    
    def getTresholdValue(self)->float:
        return self.currentTresholdValue
    
    def getHandID(self)->int:
        return self.handSelection.getCurrentHandID()
    
    def getFilePath(self)->str:
        return self.currentFilePath

    def resizeEvent(self, event):
        self.folderButton.setFixedHeight(self.folderLabel.height())


class HandSelectionWidget(Qtw.QWidget):
    changeHandSelection = pyqtSignal(int)
    def __init__(self, parent = None):
        super(HandSelectionWidget, self).__init__(parent)
        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)
        self.parent = parent

        self.layout.addWidget(Qtw.QLabel('Hand focus:'),0,0)
        self.rightCheckbox = Qtw.QCheckBox('Right')
        self.leftCheckbox = Qtw.QCheckBox('Left')
        self.layout.addWidget(self.leftCheckbox,0,1)
        self.layout.addWidget(self.rightCheckbox,0,2)

        horSpacer = Qtw.QSpacerItem(0, 0, Qtw.QSizePolicy.Expanding, Qtw.QSizePolicy.Minimum)
        self.layout.addItem(horSpacer, 0, 3)

        self.rightCheckbox.toggled.connect(lambda check: self.leftCheckbox.setChecked(not check))
        self.leftCheckbox.toggled.connect(lambda check: self.rightCheckbox.setChecked(not check))
        self.rightCheckbox.toggled.connect(lambda check: self.changeHandSelection.emit(1 if check else 0))

        self.rightCheckbox.setChecked(True)
    
    def getCurrentHandID(self):
        return 1 if self.rightCheckbox.isChecked() else 0


class DatasetController(Qtw.QWidget):
    realTimeHandDraw_Signal = pyqtSignal(bool)
    def __init__(self, parent):
        super().__init__( parent = parent)
        self.parent = parent
        self.currentFilePath = ''
        self.currentFileHeadLines = ''
        self.poseName = ''
        self.handID = 1
        self.tresholdValue = 0.0
        self.datasetList = []
        self.accuracyList = []
        self.currentDataIndex = 0
        self.datasetSaved = True

        ## Widgets initialisation
        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)

        self.fileLabel = ScrollLabel()
        self.fileLabel.setText('No file selected')
        self.fileLabel.setMaximumHeight(60)
        self.fileLabel.setMinimumWidth(180)
        self.layout.addWidget(self.fileLabel, 0,0,1,9, Qt.AlignTop)

        #self.saveButton = Qtw.QPushButton('Save dataset')
        #self.layout.addWidget(self.saveButton, 0,7,1,1, Qt.AlignTop)
        #self.saveButton.clicked.connect(self.writeDataToTxt)

        self.visuCheckbox = Qtw.QCheckBox('Visualize imported dataset')
        self.layout.addWidget(self.visuCheckbox,1,0)
        self.visuCheckbox.toggled.connect(self.visuCheckboxToggled)
        self.visuCheckbox.setEnabled(False)

        self.minusButton = Qtw.QToolButton()
        self.minusButton.setArrowType(Qt.LeftArrow)
        self.layout.addWidget(self.minusButton, 1,1,1,1)
        self.minusButton.setEnabled(False)
        self.minusButton.clicked.connect(lambda: self.setCurrentDataIndex(self.currentDataIndex-1))
        Qtw.QShortcut(Qt.Key_Left, self, lambda: self.setCurrentDataIndex(self.currentDataIndex-1))

        self.currentIndexLine = Qtw.QLineEdit(str(self.currentDataIndex))
        self.currentIndexLine.setValidator(QDoubleValidator())
        self.currentIndexLine.setMaximumWidth(25)
        self.currentIndexLine.setEnabled(False)
        self.layout.addWidget(self.currentIndexLine, 1,2,1,1)
        self.currentIndexLine.textChanged.connect(self.userIndexInput)

        self.maxIndexLabel = Qtw.QLabel(r'/0')
        self.maxIndexLabel.setEnabled(False)
        self.layout.addWidget(self.maxIndexLabel, 1,3,1,1)
        
        self.plusButton = Qtw.QToolButton()
        self.plusButton.setArrowType(Qt.RightArrow)
        self.layout.addWidget(self.plusButton, 1,4,1,1)
        self.plusButton.setEnabled(False)
        self.plusButton.clicked.connect(lambda: self.setCurrentDataIndex(self.currentDataIndex+1))
        Qtw.QShortcut(Qt.Key_Right, self, lambda: self.setCurrentDataIndex(self.currentDataIndex+1))

        self.deleteButton = Qtw.QPushButton('Delete entry')
        self.deleteButton.setEnabled(False)
        self.layout.addWidget(self.deleteButton, 1,5,1,1)
        self.deleteButton.clicked.connect(lambda: self.removeEntryDataset(self.currentDataIndex))

        self.layout.addWidget(Qtw.QLabel('Recording:'), 1,7,1,1)

        self.recordButton = SwitchButton()
        self.recordButton.setChecked(False)
        self.recordButton.setEnabled(False)
        self.layout.addWidget(self.recordButton,1,8,1,1)
        self.recordButton.clickedChecked.connect(self.startRecording)

        horSpacer = Qtw.QSpacerItem(0, 0, Qtw.QSizePolicy.Expanding, Qtw.QSizePolicy.Minimum)
        self.layout.addItem(horSpacer, 1, 6)

        verSpacer = Qtw.QSpacerItem(0, 0, Qtw.QSizePolicy.Minimum, Qtw.QSizePolicy.Expanding)
        self.layout.addItem(verSpacer, 2, 0)
    
    def createDataset(self):
        dlg = CreateDatasetDialog(self)
        if dlg.exec_():
            self.clearDataset()
            self.updateFileInfo(dlg.getFilePath(), dlg.getFileHeadlines(), 0, dlg.getPoseName(), dlg.getHandID(), dlg.getTresholdValue())
            self.setCurrentDataIndex(0)
        
    def addEntryDataset(self, keypoints, accuracy:float):
        ''' Add keypoints and accuracy of a hand pose to the local dataset.
        
        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        '''
        self.datasetList.append(keypoints)
        self.accuracyList.append(accuracy)
        self.maxIndexLabel.setText('/'+str(len(self.accuracyList)))
        self.datasetSaved = False
    
    def removeEntryDataset(self, index:int):
        ''' Remove keypoints and accuracy referenced by its index from the local dataset.
        
        Args:
            index (int): Index in list of the entry removed.
        '''
        self.datasetList = self.datasetList[:index] + self.datasetList[index+1:]
        self.accuracyList = self.accuracyList[:index] + self.accuracyList[index+1:]
        maxIndex = len(self.accuracyList)
        self.maxIndexLabel.setText('/'+str(maxIndex))
        index = min(index, maxIndex-1)
        self.setCurrentDataIndex(index)
        self.datasetSaved = False
    
    def clearDataset(self):
        self.datasetList = []
        self.accuracyList = []
        self.datasetSaved = False

    def userIndexInput(self, indexStr:str):
        if indexStr.isdigit():
            self.setCurrentDataIndex(int(indexStr)-1)
        elif len(indexStr) == 0:
            pass
        else:
            self.currentIndexLine.setText(str(self.currentDataIndex + 1))

    def visuCheckboxToggled(self, state:bool):
        self.realTimeHandDraw_Signal.emit(not state)
        self.plusButton.setEnabled(state)
        self.minusButton.setEnabled(state)
        self.currentIndexLine.setEnabled(state)
        self.maxIndexLabel.setEnabled(state)
        self.deleteButton.setEnabled(state)
        self.setCurrentDataIndex(0)

    def loadFile(self):
        options = Qtw.QFileDialog.Options()
        fileName, _ = Qtw.QFileDialog.getOpenFileName(self,"Open dataset", r".\Datasets","Text Files (*.txt)", options=options)
        self.clearDataset()
        currentEntry = []

        if fileName:
            self.clearDataset()
            dataFile = open(fileName)
            fileHeadline = ''
            for i, line in enumerate(dataFile):
                if i == 0:
                    info = line.split(',')
                    fileHeadline += line
                    if len(info) == 3:
                        poseName = info[0]
                        handID = int(info[1])
                        tresholdValue = float(info[2])
                    else:
                        self.fileLabel.setText('Not a supported dataset')
                        break
                else:
                    if line[0] == '#' and line[1] == '#': # Commentary/headlines
                        fileHeadline += line
                    elif line[0] == '#' and line[1] != '#': # New entry
                        currentEntry = [[], [], []]
                        accuracy = float(line[1:])
                    elif line[0] == 'x':
                        listStr = line[2:].split(' ')
                        for value in listStr:
                            currentEntry[0].append(float(value))
                    elif line[0] == 'y':
                        listStr = line[2:].split(' ')
                        for value in listStr:
                            currentEntry[1].append(float(value))
                    elif line[0] == 'a':  # Last line of entry
                        listStr = line[2:].split(' ')
                        for value in listStr:
                            currentEntry[2].append(float(value))
                        self.addEntryDataset(currentEntry, accuracy)

            dataFile.close()
            self.updateFileInfo(fileName, fileHeadline, len(self.datasetList), poseName, handID, tresholdValue)
            self.recordButton.setEnabled(True)
            self.visuCheckbox.setChecked(True)
            self.datasetSaved = True
            return True
        return False
    
    def updateFileInfo(self, filePath:str=None, fileHead:str=None, sizeData:int = 0, poseName:str=None, handID:int=None, tresholdValue:int=None):
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
        self.fileLabel.setText(self.currentFilePath + '\n  -> {} entries for {} ({} hand) with a minimum accuracy of {}.'.format(str(sizeData), poseName, ('right' if handID==1 else 'left'), str(tresholdValue)))
        self.maxIndexLabel.setText('/'+str(sizeData))
        self.recordButton.setEnabled(True)

    def setCurrentDataIndex(self, index:int):
        if len(self.datasetList) == 0:
            self.currentDataIndex = 0
            self.parent.leftHandAnalysis.drawHand(None, 0.0)
            self.parent.rightHandAnalysis.drawHand(None, 0.0)
        else:
            if index >= len(self.datasetList):
                index = 0
            if index < 0:
                index = len(self.datasetList)-1
            self.currentDataIndex = index

            if self.handID == 0:
                self.parent.leftHandAnalysis.drawHand(np.array(self.datasetList[self.currentDataIndex]), self.accuracyList[self.currentDataIndex])
            else:
                self.parent.rightHandAnalysis.drawHand(np.array(self.datasetList[self.currentDataIndex]), self.accuracyList[self.currentDataIndex])
        self.currentIndexLine.setText(str(self.currentDataIndex + 1))
        
    def writeDataToTxt(self):
        ''' Save the current dataset to the text file (URL: self.currentFilePath).'''
        if os.path.isfile(self.currentFilePath):
            dataFile = open(self.currentFilePath, 'w') #Open in write 'w' to clear.
            dataFile.write(self.currentFileHeadLines)
            sizeData = len(self.datasetList)
            for entryIndex in range(sizeData):
                dataFile.write('#' + str(self.accuracyList[entryIndex]))
                for i,row in enumerate(self.datasetList[entryIndex]):
                    for j,val in enumerate(row):
                        dataFile.write('\n{}:'.format(['x','y','a'][i]) if j == 0 else ' ')
                        dataFile.write(str(val))
                dataFile.write('\n\n')
            self.updateFileInfo(sizeData=sizeData)
            self.datasetSaved = True
    
    def startRecording(self, state:bool):
        self.parent.isRecording = state
    
    def getTresholdValue(self)->float:
        return self.tresholdValue
    
    def getHandID(self)->int:
        return self.handID
    
    def getPoseName(self)->str:
        return self.poseName
    
    def isSaved(self)->bool:
        return self.datasetSaved


class HandAnalysis(Qtw.QGroupBox):
    def __init__(self, handID:int, showInput:bool=True):
        super().__init__(('Right' if handID == 1 else 'left') + ' hand analysis')

        self.handID = handID
        self.showInput = showInput
        self.classOutputs = []
        self.modelClassifier = None
        self.currentPrediction = ''

        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)

        self.classGraphWidget = pg.PlotWidget()
        self.classGraphWidget.setBackground('w')
        self.classGraphWidget.setYRange(0.0, 1.0)
        self.classGraphWidget.setTitle('Predicted class: ')

        self.outputGraph = pg.BarGraphItem(x=range(len(self.classOutputs)), height=[0]*len(self.classOutputs), width=0.6, brush='k')
        self.classGraphWidget.addItem(self.outputGraph)

        if self.showInput:
            self.handGraphWidget = pg.PlotWidget()
            self.handGraphWidget.setBackground('w')
            self.handGraphWidget.setXRange(-1.0, 1.0)
            self.handGraphWidget.setYRange(-1.0, 1.0)
            self.handGraphWidget.setAspectLocked(True)

            self.graphSplitter = Qtw.QSplitter(Qt.Vertical)
            self.graphSplitter.addWidget(self.handGraphWidget)
            self.graphSplitter.addWidget(self.classGraphWidget)
            self.graphSplitter.setStretchFactor(0,2)
            self.graphSplitter.setStretchFactor(1,1)

            self.layout.addWidget(self.graphSplitter)
        else:
            self.layout.addWidget(self.classGraphWidget)
    
    def setClassifierModel(self, model:tf.keras.models, classOutputs):
        self.modelClassifier = model
        self.classOutputs = classOutputs
    
    def drawHand(self, handKeypoints:np.ndarray, accuracy:float):
        ''' Draw keypoints of a hand pose in the widget if showInput==True.
        
        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        '''
        if self.showInput:
            self.handGraphWidget.clear()
            self.handGraphWidget.setTitle('Detection accuracy: ' + str(accuracy))

            self.updatePredictedClass(handKeypoints)
            if isHandData(handKeypoints):

                colors = ['r','y','g','b','m']
                data = [handKeypoints[:, 0:5],
                        np.insert(handKeypoints[:, 5:9].T, 0, handKeypoints[:,0], axis=0).T,
                        np.insert(handKeypoints[:, 9:13].T, 0, handKeypoints[:,0], axis=0).T,
                        np.insert(handKeypoints[:, 13:17].T, 0, handKeypoints[:,0], axis=0).T,
                        np.insert(handKeypoints[:, 17:21].T, 0, handKeypoints[:,0], axis=0).T]
                for i in range(len(data)):
                    self.handGraphWidget.plot(data[i][0], data[i][1], symbol='o', symbolSize=7, symbolBrush=(colors[i]))
    
    def updatePredictedClass(self, keypoints:np.ndarray):
        ''' Draw keypoints of a hand pose in the widget.
        
        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
        '''

        prediction = [0]*len(self.classOutputs)
        title = 'Predicted class: None'
        if type(keypoints) != type(None):
            inputData = []
            for i in range(keypoints.shape[1]):
                inputData.append(keypoints[0,i]) #add x
                inputData.append(keypoints[1,i]) #add y
            inputData = np.array(inputData)

            if self.modelClassifier is not None:
                prediction = self.modelClassifier.predict(np.array([inputData]))[0]
                self.currentPrediction = self.classOutputs[np.argmax(prediction)]
                title = 'Predicted class: ' + self.currentPrediction

        self.outputGraph.setOpts(height=prediction)
        self.classGraphWidget.setTitle(title)
    
    def newModelLoaded(self, urlModel:str, classOutputs:list, handID:int):
        if urlModel == 'None':
            self.modelClassifier = None
            self.classOutputs = []
            self.outputGraph.setOpts(x=range(1,len(self.classOutputs)+1), height=[0]*len(self.classOutputs))
        
        else:
            if handID == self.handID:
                self.modelClassifier = tf.keras.models.load_model(urlModel)
                self.classOutputs = classOutputs
                self.outputGraph.setOpts(x=range(1,len(self.classOutputs)+1), height=[0]*len(self.classOutputs))
                self.modelClassifier = tf.keras.models.load_model(urlModel)
                self.classOutputs = classOutputs
                self.outputGraph.setOpts(x=range(1,len(self.classOutputs)+1), height=[0]*len(self.classOutputs))
        
    def getCurrentPrediction(self)->str:
        return self.currentPrediction

class TrainingWidget(Qtw.QMainWindow):
    def __init__(self, parent = None):
        ## Init
        super(TrainingWidget, self).__init__(parent)
        self.setWindowTitle("Hand pose classifier")
        self.parent = parent
        mainWidget = Qtw.QWidget(self)
        self.setCentralWidget(mainWidget)
        self.layout=Qtw.QGridLayout(mainWidget)
        self.layout.setColumnStretch(0,0)
        self.layout.setColumnStretch(1,0)
        self.layout.setColumnStretch(2,3)
        self.layout.setColumnStretch(3,3)
        
        mainWidget.setLayout(self.layout)

        ## Parameters
        self.isRecording = False
        self.realTimeHandDraw = True
 
        ## Widgets
        self.cameraInput = CameraInput()

        self.videoViewer = VideoViewer(self.cameraInput.getAvailableCam())
        self.videoViewer.camera_selector.currentIndexChanged.connect(self.cameraInput.select_camera)
        self.videoViewer.refreshButton.clicked.connect(self.refreshCameraList)
        self.layout.addWidget(self.videoViewer,0,0,1,2)
        
        self.datasetController = DatasetController(self)
        self.layout.addWidget(self.datasetController,1,0,1,1)
        self.datasetController.realTimeHandDraw_Signal.connect(self.changeHandDrawingState)

        videoHeight = 480 # 480p
        self.AnalysisThread = VideoAnalysisThread(self.cameraInput)
        self.AnalysisThread.newPixmap.connect(self.videoViewer.setImage)
        self.AnalysisThread.newPixmap.connect(self.analyseNewImage)
        self.AnalysisThread.setResolutionStream(int(videoHeight * (16.0/9.0)), videoHeight)
        self.videoViewer.setVideoSize(int(videoHeight * (16.0/9.0)), videoHeight)

        self.AnalysisThread.start()
        self.AnalysisThread.setState(True)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        self.graphWidget.setXRange(-1.0, 1.0)
        self.graphWidget.setYRange(-1.0, 1.0)
        #self.graphWidget.setMinimumSize(videoHeight,videoHeight)
        self.graphWidget.setAspectLocked(True)
        #self.layout.addWidget(self.graphWidget, 0,1,2,1)

        self.classifierWidget = PoseClassifierWidget(self)
        self.layout.addWidget(self.classifierWidget,1,1,1,1)

        self.leftHandAnalysis = HandAnalysis(0)
        self.classifierWidget.newClassifierModel_Signal.connect(self.leftHandAnalysis.newModelLoaded)
        self.layout.addWidget(self.leftHandAnalysis, 0,2,2,1)

        self.rightHandAnalysis = HandAnalysis(1)
        self.classifierWidget.newClassifierModel_Signal.connect(self.rightHandAnalysis.newModelLoaded)
        self.layout.addWidget(self.rightHandAnalysis, 0,3,2,1)

        ## Menu

        bar = self.menuBar()
        fileAction = bar.addMenu("Dataset")

        openAct = Qtw.QAction('&Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open dataset')
        openAct.triggered.connect(self.datasetController.loadFile)
        fileAction.addAction(openAct)

        initAct = Qtw.QAction('&Create new ...', self)
        initAct.setShortcut('Ctrl+N')
        initAct.setStatusTip('Create dataset')
        initAct.triggered.connect(self.datasetController.createDataset)
        fileAction.addAction(initAct)
        
        saveAct = Qtw.QAction('&Save', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('Save dataset')
        saveAct.triggered.connect(self.datasetController.writeDataToTxt)
        fileAction.addAction(saveAct)

        ## Status Bar

        self.statusBar = Qtw.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.openposeStatus = 'OpenPose running.' if OPENPOSE_LOADED else 'OpenPose not found'
        self.openposeStatusLabel = Qtw.QLabel('<span style="color:' + ('green' if OPENPOSE_LOADED else 'red') + '">' + self.openposeStatus + '</span>')
        self.statusBar.addWidget(self.openposeStatusLabel)

        self.tfStatus = ('TensorFlow running: ' + str(len(GPU_LIST)) + " Physical GPUs, " + str(len(logical_gpus)) + " Logical GPUs") if TF_LOADED else 'TensorFlow not found.'
        self.tfStatusLabel = Qtw.QLabel('<span style="color:' + ('green' if TF_LOADED else 'red') + '">' + self.tfStatus + '</span>')
        self.statusBar.addWidget(self.tfStatusLabel)

    def closeEvent(self, event):
        print('Closing')
        exitBool = True
        if self.datasetController.isSaved():
            exitBool = True
        else:
            reply = QMessageBox.question(self, 'Hand pose classifier',
                "Do you want to save " + self.datasetController.getPoseName() + ' dataset?', buttons = QMessageBox.StandardButtons(QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel))

            if reply == QMessageBox.Cancel:
                exitBool = False
            elif reply == QMessageBox.No:
                exitBool = True
            elif reply == QMessageBox.Yes:
                self.datasetController.writeDataToTxt()
                exitBool = True
                
        if exitBool:
            event.accept()
            self.AnalysisThread.terminate()
            #self.cameraInput.terminate()
            time.sleep(1.0)
            print(self.cameraInput.deleteTmpImage())
        else:
            event.ignore()
    
    def keyPressEvent(self, event):
        print(event.key())
        print(self.datasetController.deleteButton.isEnabled())
        if event.key() == 16777223 and self.datasetController.deleteButton.isEnabled():
            self.datasetController.removeEntryDataset(self.datasetController.currentDataIndex)
    
    def refreshCameraList(self):
        camList = self.cameraInput.refreshCameraList()
        if not camList:
            print('No camera')
        else:
            self.videoViewer.camera_selector.clear()
            self.videoViewer.camera_selector.addItems([c.description() for c in camList])

    def analyseNewImage(self, image): # Call each time AnalysisThread emit a new pix
        self.videoViewer.setInfoText(self.AnalysisThread.getInfoText())
        
        leftHandKeypoints, leftAccuracy = self.AnalysisThread.getHandData(0)
        rightHandKeypoints, rightAccuracy = self.AnalysisThread.getHandData(1)
        poseKeypoints = self.AnalysisThread.getBodyData()
        raisingLeft, raisingRight = self.AnalysisThread.isRaisingHand()
        #print('Gauche: ' + str(raisingLeft))
        #print('Droite: ' + str(raisingRight))

        if self.realTimeHandDraw:
            self.leftHandAnalysis.drawHand(leftHandKeypoints, leftAccuracy)
            self.rightHandAnalysis.drawHand(rightHandKeypoints, rightAccuracy)
        
        if self.datasetController.getHandID() == 0: # Recording left hand
            if type(leftHandKeypoints) != type(None):
                if self.isRecording:
                    if leftAccuracy > self.datasetController.getTresholdValue():
                        self.datasetController.addEntryDataset(leftHandKeypoints, leftAccuracy)
        else: # Recording right hand
            if type(rightHandKeypoints) != type(None): # If selected hand detected
                if self.isRecording:
                    if rightAccuracy > self.datasetController.getTresholdValue():
                        self.datasetController.addEntryDataset(rightHandKeypoints, rightAccuracy)

    def changeHandDrawingState(self, state:bool):
        self.realTimeHandDraw = state
    

class PoseClassifierWidget(Qtw.QWidget):
    newClassifierModel_Signal = pyqtSignal(str, list, int) # url to load classifier model, output labels, handID
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.modelRight=None
        self.modelLeft=None

        self.classOutputs = []
        self.leftWidget = Qtw.QWidget()
        self.layout=Qtw.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0)
        classifierLabel = Qtw.QLabel('Classifier:')
        classifierLabel.setSizePolicy(Qtw.QSizePolicy.Minimum, Qtw.QSizePolicy.Minimum)
        self.layout.addWidget(classifierLabel,1,0,1,1)
        
        self.classifierSelector = Qtw.QComboBox()
        #self.classifierSelector.setSizePolicy(Qtw.QSizePolicy.Expanding, Qtw.QSizePolicy.Expanding)
        self.classifierSelector.addItems(self.getAvailableClassifiers())
        self.layout.addWidget(self.classifierSelector,1,1,1,1)
        self.classifierSelector.currentTextChanged.connect(self.loadModel)

        updateClassifierButton = Qtw.QPushButton('Update list')
        updateClassifierButton.clicked.connect(self.updateClassifier)
        self.layout.addWidget(updateClassifierButton,1,2,1,1)

        self.tableWidget = Qtw.QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['Class'])
        #self.tableWidget.setEnabled(False)
        self.tableWidget.setEditTriggers(Qtw.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setFocusPolicy(Qt.NoFocus)
        self.tableWidget.setSelectionMode(Qtw.QAbstractItemView.NoSelection)
        self.layout.addWidget(self.tableWidget,0,0,1,3)


    def loadModel(self, name:str):
        ''' Load full (structures + weigths) h5 model.
        
            Args:
                name (string): Name of the model. The folder .\models\name must contain: modelName_right.h5, modelName_left.h5, class.txt
        '''
        if name != 'None':
            urlFolder = r'.\Models' + '\\' + name
            if os.path.isdir(urlFolder):
                urlRight = urlFolder + '\\' + name + '_right.h5'
                urlLeft = urlFolder + '\\' + name + '_left.h5'
                urlClass = urlFolder + '\\' + 'class.txt'
                if os.path.isfile(urlClass):
                    with open(urlClass, "r") as file:
                        first_line = file.readline()
                    self.classOutputs = first_line.split(',')
                    self.tableWidget.setRowCount(len(self.classOutputs))
                    for i,elem in enumerate(self.classOutputs):
                        self.tableWidget.setItem(i,0, Qtw.QTableWidgetItem(elem))
                    print('Class model loaded.')
                if os.path.isfile(urlRight):
                    self.newClassifierModel_Signal.emit(urlRight, self.classOutputs, 1)
                    print('Right hand model loaded.')
                if os.path.isfile(urlLeft):
                    self.newClassifierModel_Signal.emit(urlLeft, self.classOutputs, 0)
                    print('Left hand model loaded.')
        else:
            print('None')
            self.modelRight = None
            self.modelLeft = None
            self.classOutputs = []
            self.newClassifierModel_Signal.emit('None', [], -1)
            self.tableWidget.setRowCount(0)

    def getAvailableClassifiers(self):
        listOut = ['None']
        listOut += [name for name in os.listdir(r'.\Models') if os.path.isdir(r'.\Models\\'+name)]
        return listOut
    
    def updateClassifier(self):
        self.classifierSelector.clear()
        self.classifierSelector.addItems(self.getAvailableClassifiers())

class HandSignalDetector(Qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.classOutputs = []

        layout = Qtw.QGridLayout(self)
        self.setLayout(layout)

        self.cameraInput = CameraInput()

        self.videoViewer = VideoViewer(self.cameraInput.getAvailableCam())
        self.videoViewer.camera_selector.currentIndexChanged.connect(self.cameraInput.select_camera)
        self.videoViewer.refreshButton.clicked.connect(self.refreshCameraList)
        layout.addWidget(self.videoViewer,0,0,1,3)

        videoHeight = 480 # 480p
        self.AnalysisThread = VideoAnalysisThread(self.cameraInput, False)
        self.AnalysisThread.newMat.connect(self.analyseNewImage)
        self.AnalysisThread.setResolutionStream(int(videoHeight * (16.0/9.0)), videoHeight)
        self.AnalysisThread.start()
        self.AnalysisThread.setState(True)
        self.videoViewer.setVideoSize(int(videoHeight * (16.0/9.0)), videoHeight)

        self.leftHandAnalysis = HandAnalysis(0, showInput=False)
        self.rightHandAnalysis = HandAnalysis(1, showInput=False)

        self.tableWidget = Qtw.QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['Class'])
        self.tableWidget.setEditTriggers(Qtw.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setFocusPolicy(Qt.NoFocus)
        self.tableWidget.setSelectionMode(Qtw.QAbstractItemView.NoSelection)
        self.tableWidget.setMinimumWidth(200)
        self.tableWidget.setMaximumWidth(200)

        layout.addWidget(self.leftHandAnalysis, 1,0,1,1)
        layout.addWidget(self.rightHandAnalysis, 1,1,1,1)
        layout.addWidget(self.tableWidget,1,2,1,1)

        self.detailedView(False)

        self.loadModel('24Output-2x128-17epochs')

    
    def refreshCameraList(self):
        camList = self.cameraInput.refreshCameraList()
        if not camList:
            print('No camera')
        else:
            self.videoViewer.camera_selector.clear()
            self.videoViewer.camera_selector.addItems([c.description() for c in camList])
    
    def analyseNewImage(self, matImage:np.ndarray): # Call each time AnalysisThread emit a new pix
        self.videoViewer.setInfoText(self.AnalysisThread.getInfoText())
        
        leftHandKeypoints, leftAccuracy = self.AnalysisThread.getHandData(0)
        rightHandKeypoints, rightAccuracy = self.AnalysisThread.getHandData(1)
        poseKeypoints = self.AnalysisThread.getBodyData()
        raisingLeft, raisingRight = self.AnalysisThread.isRaisingHand()

        self.leftHandAnalysis.updatePredictedClass(leftHandKeypoints)
        self.rightHandAnalysis.updatePredictedClass(rightHandKeypoints)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (255, 0, 255)
        print('Left:' + self.leftHandAnalysis.getCurrentPrediction())
        print('Right:' + self.rightHandAnalysis.getCurrentPrediction())

        if isHandData(leftHandKeypoints):
            position = (poseKeypoints[7][0],poseKeypoints[7][1]) # (0,0) <=> left-up corner
            cv2.putText(matImage, self.leftHandAnalysis.getCurrentPrediction(), position, font, scale, color, 2, cv2.LINE_AA)
        if isHandData(rightHandKeypoints):
            position = (poseKeypoints[4][0],poseKeypoints[4][1]) # (0,0) <=> left-up corner
            cv2.putText(matImage, self.rightHandAnalysis.getCurrentPrediction(), position, font, scale, color, 2, cv2.LINE_AA)
        self.videoViewer.setImage(mat2QImage(matImage))
    
    def loadModel(self, name:str):
        ''' Load full (structures + weigths) h5 model.
        
            Args:
                name (string): Name of the model. The folder .\models\name must contain: modelName_right.h5, modelName_left.h5, class.txt
        '''
        if name != 'None':
            urlFolder = pathlib.Path(__file__).parent.absolute() / 'Models' / name
            if os.path.isdir(urlFolder):
                urlRight = urlFolder / (name + '_right.h5')
                urlLeft = urlFolder / (name + '_left.h5')
                urlClass = urlFolder / 'class.txt'
                if os.path.isfile(urlClass):
                    with open(urlClass, "r") as file:
                        first_line = file.readline()
                    self.classOutputs = first_line.split(',')
                    self.tableWidget.setRowCount(len(self.classOutputs))
                    for i,elem in enumerate(self.classOutputs):
                        self.tableWidget.setItem(i,0, Qtw.QTableWidgetItem(elem))
                    print('Class model loaded.')
                if os.path.isfile(urlRight):
                    self.rightHandAnalysis.newModelLoaded(str(urlRight), self.classOutputs, 1)
                    print('Right hand model loaded.')
                if os.path.isfile(urlLeft):
                    self.leftHandAnalysis.newModelLoaded(str(urlLeft), self.classOutputs, 0)
                    print('Left hand model loaded.')
        else:
            print('None')
            self.modelRight = None
            self.modelLeft = None
            self.classOutputs = []
            self.leftHandAnalysis.newModelLoaded('None', self.classOutputs, -1)
            self.rightHandAnalysis.newModelLoaded('None', self.classOutputs, -1)
            self.tableWidget.setRowCount(0)
    
    @pyqtSlot(bool)
    def detailedView(self, b:bool):
        if b:
            self.leftHandAnalysis.show()
            self.rightHandAnalysis.show()
            self.tableWidget.show()
        else:
            self.leftHandAnalysis.hide()
            self.rightHandAnalysis.hide()
            self.tableWidget.hide()
        


if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication
    import sys
    FULL_APP = False
    app = Qtw.QApplication(sys.argv)
    
    if FULL_APP:
        trainingWidget = TrainingWidget()
        trainingWidget.show()
    else:
        handSignalDetector = HandSignalDetector()
        handSignalDetector.show()

    sys.exit(app.exec_())