import pathlib
import cv2
import os
import time
import numpy as np
import sys

from __init__ import OPENPOSE_PATH
from .qt import QtWidgets, QtCore, QtGui, QtMultimedia, pyqtSignal, pyqtSlot
from .Util import mat2QImage


try:
    sys.path.append(str(OPENPOSE_PATH / "build" / "python" / "openpose" / "Release"))
    releasePATH = OPENPOSE_PATH / "build" / "x64" / "Release"
    binPATH = OPENPOSE_PATH / "build" / "bin"
    modelsPATH = OPENPOSE_PATH / "models"
    os.environ["PATH"] = (
        os.environ["PATH"] + ";" + str(releasePATH) + ";" + str(binPATH) + ";"
    )
    import pyopenpose as op

    OPENPOSE_LOADED = True
except:
    OPENPOSE_LOADED = False
    print("OpenPose ({}) loading failed.".format(str(OPENPOSE_PATH)))


class CameraInput(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(CameraInput, self).__init__(*args, **kwargs)

        self.available_cameras = QtMultimedia.QCameraInfo.availableCameras()

        if not self.available_cameras:
            print("No camera")
            pass  # quit

        self.buffer = QtCore.QBuffer
        # self.lastImage = QtGui.QImage('.\\Data\\tempInit.png')
        self.lastImage = QtGui.QPixmap(10, 10).toImage()
        self.lastID = None
        self.save_path = ""
        self.tmpUrl = str(
            pathlib.Path(__file__).parent.absolute() / "tmp.png"
        )  # / 'Data'

        self.capture = None

        self.select_camera(0)

    def refreshCameraList(self):
        self.available_cameras = QtMultimedia.QCameraInfo.availableCameras()
        if not self.available_cameras:
            print("No camera")
            return None
        self.camera.stop()
        self.select_camera(0)
        return self.available_cameras

    def getAvailableCam(self):
        return self.available_cameras

    def select_camera(self, i):
        if len(self.available_cameras) > 0:
            self.camera = QtMultimedia.QCamera(self.available_cameras[i])
            self.camera.setCaptureMode(QtMultimedia.QCamera.CaptureStillImage)
            self.camera.start()

            self.capture = QtMultimedia.QCameraImageCapture(self.camera)
            self.capture.setCaptureDestination(
                QtMultimedia.QCameraImageCapture.CaptureToBuffer
            )

            self.capture.imageCaptured.connect(self.storeLastFrame)

            self.current_camera_name = self.available_cameras[i].description()
            self.save_seq = 0
        else:
            print("No camera.")

    def getLastFrame(self):
        if self.capture:
            imageID = self.capture.capture()
            return self.qImageToMat(self.lastImage)
        else:
            return None

    def storeLastFrame(self, idImg: int, preview: QtGui.QImage):
        self.lastImage = preview
        self.lastID = idImg

    def qImageToMat(self, incomingImage):
        incomingImage.save(self.tmpUrl, "png")
        mat = cv2.imread(self.tmpUrl)
        return mat

    def deleteTmpImage(self):
        os.remove(self.tmpUrl)
        self.tmpUrl = None


""" No temporary files but too slow
    def qImageToMat(self, incomingImage):
        incomingImage = incomingImage.convertToFormat(4) #Set to format RGB32
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits() #Get pointer to first pixel
        ptr.setsize(height * width * 4) #Get pointer to full image
        arr = np.array(ptr).reshape(height, width, 4)  #Copies the data
        arr = np.delete(arr, 3, 2) #Delete alpha channel
        return arr
"""


class VideoViewerWidget(QtWidgets.QGroupBox):
    changeCameraID_signal = pyqtSignal

    def __init__(self, availableCameras):
        super().__init__("Camera feed")

        self.availableCameras = availableCameras

        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        self.rawCamFeed = QtWidgets.QLabel(self)

        self.infoLabel = QtWidgets.QLabel("No info")

        self.refreshButton = QtWidgets.QPushButton("Refresh camera list")

        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.addItems([c.description() for c in self.availableCameras])

        if OPENPOSE_LOADED:
            self.layout.addWidget(self.rawCamFeed, 0, 0, 1, 3)
            self.layout.addWidget(self.infoLabel, 1, 2, 1, 1)
            self.layout.addWidget(self.refreshButton, 1, 0, 1, 1)
            self.layout.addWidget(self.camera_selector, 1, 1, 1, 1)
        else:
            label = QtWidgets.QLabel(
                "Video analysis impossible.\nCheck OpenPose installation."
            )
            self.layout.addWidget(label, 0, 0, 1, 1)

        self.autoAdjustable = False

    @pyqtSlot(QtGui.QImage)
    def setImage(self, image: QtGui.QImage):
        self.currentPixmap = QtGui.QPixmap.fromImage(image)
        self.rawCamFeed.setPixmap(
            self.currentPixmap.scaled(
                self.rawCamFeed.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        if self.autoAdjustable:
            try:
                w = self.rawCamFeed.width()
                h = self.rawCamFeed.height()
                pixmap = self.currentPixmap.scaled(
                    w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                self.rawCamFeed.setPixmap(pixmap)
                self.rawCamFeed.setMinimumSize(100, 100)

                w = self.pepperCamFeed.width()
                h = self.pepperCamFeed.height()
                self.pepperCamFeed.setPixmap(
                    self.pepperCamFeed.scaled(
                        w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                    )
                )
                self.pepperCamFeed.setMinimumSize(100, 100)
            except:
                pass

    def setVideoSize(self, width: int, height: int):
        self.rawCamFeed.setFixedSize(width, height)

    def setInfoText(self, info: str):
        if info:
            self.infoLabel.setText(info)
        else:
            self.infoLabel.setText("")


class VideoAnalysisThread(QtCore.QThread):
    newPixmap = pyqtSignal(QtGui.QImage)
    newMat = pyqtSignal(np.ndarray)

    def __init__(self, videoSource, qimageEmission: bool = True):
        super().__init__()
        self.infoText = ""
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
            params["disable_multi_thread"] = False
            netRes = 15  # Default 22
            params["net_resolution"] = "-1x" + str(16 * netRes)

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
                if (
                    time.time() - self.lastTime > 1.0 / self.emissionFPS
                ) or not self.fixedFps:
                    self.lastTime = time.time()

                    frame = self.videoSource.getLastFrame()
                    if type(frame) != type(None):
                        # Check if frame exist, frame!=None is ambigious when frame is an array
                        frame = self.resizeCvFrame(frame, 0.5)
                        self.datum.cvInputData = frame
                        self.opWrapper.emplaceAndPop([self.datum])
                        frameOutput = self.datum.cvOutputData
                        self.newMat.emit(frameOutput)

                        if self.qimageEmission:
                            image = mat2QImage(frameOutput)
                            image = image.scaled(
                                self.videoWidth,
                                self.videoHeight,
                                QtCore.Qt.KeepAspectRatio,
                            )
                            self.newPixmap.emit(image)

    @pyqtSlot(bool)
    def setState(self, s: bool):
        self.running = s

    def setResolutionStream(self, width: int, height: int):
        self.videoHeight = height
        self.videoWidth = width

    def setEmissionSpeed(self, fixedFPS: bool, fps: int):
        self.fixedFps = fixedFPS
        if self.fixedFps:
            self.emissionFPS = fps

    def getHandData(self, handID: int):
        """Return the key points of the hand seen in the image (cf. videoSource).

        Args:
            handID (int): 0 -> Left hand | 1 -> Right hand

        Returns:
            np.ndarray((3,21),float): Coordinates x, y and the accuracy score for each 21 key points.
                                      None if the given hand is not detected.
        """
        outputArray = None

        handKeypoints = np.array(self.datum.handKeypoints)
        nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim > 2 else 0
        handAccuaracyScore = 0.0
        if nbrPersonDetected > 0:
            handAccuaracyScore = handKeypoints[handID, self.personID].T[2].sum()
            handDetected = handAccuaracyScore > 1.0
            if handDetected:
                handKeypoints = handKeypoints[handID, self.personID]
                # Initialize with the length of the first segment of each fingers
                lengthFingers = [
                    np.sqrt(
                        (handKeypoints[0, 0] - handKeypoints[i, 0]) ** 2
                        + (handKeypoints[0, 1] - handKeypoints[i, 1]) ** 2
                    )
                    for i in [1, 5, 9, 13, 17]
                ]
                for i in range(3):  # Add length of other segments of each fingers
                    for j in range(len(lengthFingers)):
                        x = (
                            handKeypoints[1 + j * 4 + i + 1, 0]
                            - handKeypoints[1 + j * 4 + i, 0]
                        )
                        y = (
                            handKeypoints[1 + j * 4 + i + 1, 1]
                            - handKeypoints[1 + j * 4 + i, 1]
                        )
                        lengthFingers[j] += np.sqrt(x ** 2 + y ** 2)
                normMax = max(lengthFingers)

                handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
                handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]
                outputArray = np.array(
                    [
                        (handKeypoints.T[0] - handCenterX) / normMax,
                        -(handKeypoints.T[1] - handCenterY) / normMax,
                        (handKeypoints.T[2]),
                    ]
                )
        return outputArray, handAccuaracyScore

    def getBodyData(self):
        if len(self.datum.poseKeypoints.shape) > 0:
            poseKeypoints = self.datum.poseKeypoints[self.personID]
            return poseKeypoints
        else:
            return None

    def getInfoText(self) -> str:
        handKeypoints = np.array(self.datum.handKeypoints)
        nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim > 2 else 0

        self.infoText = ""
        self.infoText += str(nbrPersonDetected) + (
            " person detected" if nbrPersonDetected < 2 else " person detected"
        )

        if nbrPersonDetected > 0:
            leftHandDetected = handKeypoints[0, self.personID].T[2].sum() > 1.0
            rightHandDetected = handKeypoints[1, self.personID].T[2].sum() > 1.0
            if rightHandDetected and leftHandDetected:
                self.infoText += (
                    ", both hands of person " + str(self.personID + 1) + " detected."
                )
            elif rightHandDetected or leftHandDetected:
                self.infoText += (
                    ", "
                    + ("Right" if rightHandDetected else "Left")
                    + " hand of person "
                    + str(self.personID + 1)
                    + " detected."
                )
            else:
                self.infoText += (
                    ", no hand of person " + str(self.personID + 1) + " detected."
                )

        return self.infoText

    def getFingerLength(self, fingerData):
        length = 0.0
        for i in range(fingerData.shape[0] - 1):
            x = fingerData[i + 1, 0] - fingerData[i, 0]
            y = fingerData[i + 1, 1] - fingerData[i, 1]
            length += np.sqrt(x ** 2 + y ** 2)
        return length

    def resizeCvFrame(self, frame, ratio: float):
        width = int(frame.shape[1] * ratio)
        height = int(frame.shape[0] * ratio)
        dim = (width, height)
        # resize image in down scale
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
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

            try:
                shoulderSlope = (rightShoulder_y - leftShoulder_y) / (
                    rightShoulder_x - leftShoulder_x
                )
            except:
                shoulderSlope = 0.0
            shoulderOri = rightShoulder_y - shoulderSlope * rightShoulder_x

            if leftHand_a > 0.1:
                raisingLeft = leftHand_y < (
                    shoulderSlope * leftHand_x + shoulderOri
                )  # y axis oriented from top to down in images
                raisingLeft = (
                    raisingLeft and leftHand_y < poseKeypoints[6, 1]
                )  # Check if hand above elbow
            else:
                raisingLeft = False

            if rightHand_a > 0.1:
                raisingRight = rightHand_y < (shoulderSlope * rightHand_x + shoulderOri)
                raisingRight = raisingRight and rightHand_y < poseKeypoints[3, 1]
            else:
                raisingRight = False

        return raisingLeft, raisingRight
