from .qt import QtWidgets, QtCore, QtGui, QtMultimedia, pyqtSignal, pyqtSlot
from .openpose_thread import OPENPOSE_LOADED

import pathlib
import cv2
import numpy as np
import qimage2ndarray


class CameraInput:
    def __init__(self):
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
            frame = self.qImageToMat(self.lastImage.mirrored())
            return frame
        else:
            return None

    def storeLastFrame(self, idImg: int, preview: QtGui.QImage):
        self.lastImage = preview
        self.lastID = idImg

    def qImageToMat(self, incomingImage):
        # qimage2ndarray not working...
        incomingImage.save(self.tmpUrl, "png")
        mat = cv2.imread(self.tmpUrl)
        return mat

class ImageWidget(QtWidgets.QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setScaledContents(True)
        self.setMinimumWidth(100)

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, w):
        if self.pixmap():
            return int(w * (self.pixmap().height() / self.pixmap().width()))

class VideoViewerWidget(QtWidgets.QWidget):
    changeCameraID_signal = pyqtSignal()
    stylesheet = """
    #Video_viewer {
        background-color: white;
        border-radius: 3px;
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
    #OpenPose_button {
        border: 1px solid #cbcbcb;
        border-radius: 2px;
        font-size: 16px;
        background: #ffcccc;
        padding: 3px;
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
        background: #cbcbcb;
    }
    #OpenPose_button:checked {
        background: #ccffcc;
    }
    """

    def __init__(self, availableCameras):
        super().__init__()
        self.availableCameras = availableCameras

        ## Widget style
        self.setObjectName("Video_viewer")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(self.stylesheet)

        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(10)
        effect.setOffset(0, 0)
        effect.setColor(QtCore.Qt.gray)
        self.setGraphicsEffect(effect)

        ## Widgets initialisation
        self.cameraFeed = ImageWidget(self)

        self.infoLabel = QtWidgets.QLabel(
            "No info"
            if OPENPOSE_LOADED
            else "Video analysis impossible, check OpenPose installation."
        )

        self.refreshButton = QtWidgets.QPushButton(
            "Refresh camera list",
            cursor=QtCore.Qt.PointingHandCursor,
            toolTip="Update available camera list",
        )
        self.refreshButton.resize(self.refreshButton.sizeHint())

        self.camera_selector = QtWidgets.QComboBox(cursor=QtCore.Qt.PointingHandCursor)
        self.camera_selector.addItems([c.description() for c in self.availableCameras])

        ## Widget structure
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 0)
        self.layout.setColumnStretch(2, 0)
        self.layout.setColumnStretch(3, 1)

        if OPENPOSE_LOADED:
            self.layout.addWidget(self.cameraFeed, 0, 0, 1, 5)
            self.layout.addWidget(self.refreshButton, 1, 0, 1, 1)
            self.layout.addWidget(self.camera_selector, 1, 1, 1, 1)
            self.layout.addWidget(self.infoLabel, 1, 3, 1, 1)
        else:
            label = QtWidgets.QLabel(
                "Video analysis impossible.\nCheck OpenPose installation."
            )
            self.layout.addWidget(label, 0, 0, 1, 1)
    
    @pyqtSlot(np.ndarray)
    def setFrame(self, frame: np.ndarray):
        image = qimage2ndarray.array2qimage(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
        self.currentPixmap = QtGui.QPixmap.fromImage(image)
        self.cameraFeed.setPixmap(
            self.currentPixmap.scaled(
                self.cameraFeed.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    #def setVideoSize(self, width: int, height: int):
    #    self.cameraFeed.setFixedSize(width, height)

    def setInfoText(self, info: str):
        if info:
            self.infoLabel.setText(info)
        else:
            self.infoLabel.setText("")
