import cv2
import numpy as np

from PyQt5 import QtWidgets as Qtw
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, pyqtSignal

def isHandData( keypoints):
    b = False
    if type(keypoints) == np.ndarray:
        if keypoints.shape == (3,21):
            b = True
    return b

def mat2QImage(frame:np.ndarray)->QtGui.QImage:
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
    return convertToQtFormat

def printHeadLine(name:str = '', mainTitle:bool = True, length:int = 80):
    '''Print title in console:
        #### TITLE ####
        ###############

    Parameters:
        name (str): Title displayed
        mainTitle(bool): Add second '#' line if True
        length(int): Length of lines
    '''

    print('')
    length = max(length,len(name))
    if len(name) > 0:
        firstLine = '#'*((length-len(name))//2) + ' ' + name + ' ' + '#'*((length-len(name))//2)
        print(firstLine)
        if mainTitle:
            print('#'*len(firstLine))
    else:
        print('#'*length)
    print('')

def resizeCvFrame(frame, ratio:float):
    width = int(frame.shape[1] * ratio) 
    height = int(frame.shape[0] * ratio) 
    dim = (width, height) 
    # resize image in down scale
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
    return resized

class SwitchButton(Qtw.QPushButton):
    clickedChecked = pyqtSignal(bool)

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)
        self.clicked.connect(self.click)

    def paintEvent(self, event):
        label = "ON" if self.isChecked() else "OFF"
        if self.isEnabled():
            bg_color = Qt.green if self.isChecked() else Qt.red
        else:
            bg_color = Qt.gray

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0,0,0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(QRect(-width, -radius, 2*width, 2*radius), radius, radius)
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2*radius)
        if not self.isChecked():
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.drawText(sw_rect, Qt.AlignCenter, label)
    
    def click(self):
        b = self.isChecked()
        self.clickedChecked.emit(b)
    
    #def setEnabled(self):


class ScrollLabel(Qtw.QScrollArea): 
    def __init__(self):
        super().__init__()

        self.setWidgetResizable(True) 
        content = Qtw.QWidget(self) 
        self.setWidget(content) 
        lay = Qtw.QVBoxLayout(content) 
        self.label = Qtw.QLabel(content) 
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop) 
        lay.addWidget(self.label) 

    def setText(self, text): 
        self.label.setText(text) 

def euler_to_quaternion(roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

class VLine(Qtw.QFrame):
    # a simple VLine, like the one you get from designer
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine|self.Sunken)