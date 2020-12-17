from .qt import QtCore, QtGui, pyqtSignal, pyqtSlot
from __init__ import OPENPOSE_PATH

import cv2
import numpy as np
import os
import sys

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


class VideoAnalysisThread(QtCore.QThread):
    newFrame = pyqtSignal(np.ndarray)

    def __init__(self, videoSource):
        super().__init__()
        self.infoText = ""
        self.personID = 0
        self.running = False
        self.last_frame = np.array([])
        self.videoSource = videoSource

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

    def run(self):
        while OPENPOSE_LOADED:
            if self.running:
                frame = self.videoSource.getLastFrame()
                if (type(frame) != type(None)) and not np.array_equal(
                    self.last_frame, frame
                ):
                    self.last_frame = frame
                    # Check if frame exist, frame!=None is ambigious when frame is an array
                    frame = self.resizeCvFrame(frame, 0.5)
                    self.datum.cvInputData = frame
                    self.opWrapper.emplaceAndPop([self.datum])
                    frameOutput = self.datum.cvOutputData
                    self.newFrame.emit(frameOutput)

    @pyqtSlot(bool)
    def setState(self, s: bool):
        self.running = s

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
