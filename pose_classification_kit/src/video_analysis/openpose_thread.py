from ..imports.qt import QtCore, pyqtSignal, pyqtSlot
from ..imports.openpose import OPENPOSE_LOADED, OPENPOSE_MODELS_PATH
if OPENPOSE_LOADED:
    from ..imports.openpose import op
import cv2
import numpy as np


def getLengthLimb(data, keypoint1: int, keypoint2: int):
    if data[keypoint1, 2] > 0.0 and data[keypoint2, 2] > 0:
        return np.linalg.norm([data[keypoint1, 0:2] - data[keypoint2, 0:2]])
    return 0


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
            params["model_folder"] = str(OPENPOSE_MODELS_PATH)
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

        outputArray = None
        accuaracyScore = 0.0
        if len(self.datum.poseKeypoints.shape) > 0:

            # Read body data
            outputArray = self.datum.poseKeypoints[self.personID]
            accuaracyScore = outputArray[:, 2].sum()

            # Find bouding box
            min_x, max_x = float("inf"), 0.0
            min_y, max_y = float("inf"), 0.0
            for keypoint in outputArray:
                if keypoint[2] > 0.0:  # If keypoint exists in image
                    min_x = min(min_x, keypoint[0])
                    max_x = max(max_x, keypoint[0])
                    min_y = min(min_y, keypoint[1])
                    max_y = max(max_y, keypoint[1])

            # Centering
            np.subtract(
                outputArray[:, 0],
                (min_x + max_x) / 2,
                where=outputArray[:, 2] > 0.0,
                out=outputArray[:, 0],
            )
            np.subtract(
                (min_y + max_y) / 2,
                outputArray[:, 1],
                where=outputArray[:, 2] > 0.0,
                out=outputArray[:, 1],
            )

            # Scaling
            normalizedPartsLength = np.array(
                [
                    getLengthLimb(outputArray, 1, 8) * (16.0 / 5.2),  # Torso
                    getLengthLimb(outputArray, 0, 1) * (16.0 / 2.5),  # Neck
                    getLengthLimb(outputArray, 9, 10) * (16.0 / 3.6),  # Right thigh
                    getLengthLimb(outputArray, 10, 11)
                    * (16.0 / 3.5),  # Right lower leg
                    getLengthLimb(outputArray, 12, 13) * (16.0 / 3.6),  # Left thigh
                    getLengthLimb(outputArray, 13, 14) * (16.0 / 3.5),  # Left lower leg
                    getLengthLimb(outputArray, 2, 5) * (16.0 / 3.4),  # Shoulders
                ]
            )

            # Mean of non-zero values
            scaleFactor = np.mean(normalizedPartsLength[normalizedPartsLength > 0.0])
            if scaleFactor == 0:
                print("Scaling error")
                return None, 0.0

            np.divide(outputArray[:, 0:2], scaleFactor, out=outputArray[:, 0:2])

            if np.any((outputArray > 1.0) | (outputArray < -1.0)):
                print("Scaling error")
                return None, 0.0

            outputArray = outputArray.T

        return outputArray, accuaracyScore

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
