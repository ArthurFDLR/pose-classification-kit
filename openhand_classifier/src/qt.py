import sys

PYSIDE2_LOADED = False
PYQT5_LOADED = False

if not PYSIDE2_LOADED:
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore, QtMultimedia
        from PyQt5.QtCore import pyqtSignal, pyqtSlot
        print('Use PyQt5')
        PYQT5_LOADED = True
    except:
        pass

if not PYQT5_LOADED:
    try:
        from PySide2 import QtGui, QtWidgets, QtCore, QtMultimedia
        from PySide2.QtCore import Signal as pyqtSignal, Slot as pyqtSlot
        print('Use PySide2')
        PYSIDE2_LOADED = True
    except:
        pass
