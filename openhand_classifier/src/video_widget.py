#from .qt import QtWidgets, QtCore, QtGui, QtMultimedia, pyqtSignal
from openpose_analysis import VideoAnalysis

from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import qimage2ndarray
import numpy as np
import cv2

class VideoThread(QtCore.QObject):
    new_raw_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame_BGR = cap.read()
            if ret:
                self.new_raw_frame.emit(frame_BGR)

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
    
class VideoViewerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(VideoViewerWidget, self).__init__(parent)

        ## Variables and initialization
        self._run_analysis = True
        self.displayed_frame = np.array([])

        ## Widget Style
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Background, QtCore.Qt.white)
        self.setPalette(palette)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)

        ## Video thread initialization
        self.video_thread = QtCore.QThread()
        self.video_worker = VideoThread()
        self.video_worker.moveToThread(self.video_thread)

        self.video_worker.new_raw_frame.connect(self.store_last_frame)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_thread.finished.connect(self.video_worker.stop)
        self.video_thread.finished.connect(self.video_worker.deleteLater)
        self.video_thread.finished.connect(self.video_thread.deleteLater)

        self.video_thread.start()

        ## Analysis thread initialization
        self.analysis_thread = QtCore.QThread()
        self.analysis_worker = VideoAnalysis()
        self.analysis_worker.setState(self._run_analysis)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.analysis_thread.started.connect(self.analysis_worker.run_analysis)
        self.analysis_thread.finished.connect(self.analysis_worker.stop)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_worker.new_analysed_frame.connect(self.display_frame)
        
        self.analysis_thread.start()

    @pyqtSlot(np.ndarray)
    def store_last_frame(self, new_frame:np.ndarray): #HWC (BGR) format
        self.analysis_worker.update_input_frame(new_frame)
        if self._run_analysis:
            self.analysis_worker.update_input_frame(new_frame)
        else:
            self.display_frame(new_frame)
    
    @pyqtSlot(np.ndarray)
    def display_frame(self, new_frame:np.ndarray):
        self.displayed_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if (self.video_thread.isRunning() and self.displayed_frame.ndim == 3):
            frame_size = QtCore.QSize(self.displayed_frame.shape[1], self.displayed_frame.shape[0])
            videoRect = QtCore.QRect(QtCore.QPoint(0, 0), frame_size)
            if not videoRect.contains(event.rect()):
                region = event.region()
                region_sub = region.subtracted(QtGui.QRegion(videoRect))
                brush = self.palette().window()
                for rect in region_sub.rects():
                    painter.fillRect(rect, brush)

            image = qimage2ndarray.array2qimage(self.displayed_frame)

            frame_size.scale(self.size(), QtCore.Qt.KeepAspectRatio)
            target_rect = QtCore.QRect(QtCore.QPoint(0, 0), frame_size)
            target_rect.moveCenter(self.rect().center())

            painter.drawImage(target_rect, image)
        else:
            painter.fillRect(event.rect(), QtCore.Qt.white)

    def closeEvent(self, event):
        if self.video_thread.isRunning():
            self.video_thread.quit()
        super(VideoViewerWidget, self).closeEvent(event)
        event.accept()


#poetry run python .\openhand_classifier\src\video_widget.py
if __name__ == '__main__':
    import sys
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)
            self.videoWidget = VideoViewerWidget()
            self.setCentralWidget(self.videoWidget)
            self.setWindowTitle("WebCam")
            self.show()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("WebCam")

    window = MainWindow()
    app.exec_()

