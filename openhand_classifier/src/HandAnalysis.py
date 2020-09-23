import numpy as np
import os

from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib import figure, lines, patches, path

from .qt import QtWidgets, QtCore, \
    PYSIDE2_LOADED, PYQT5_LOADED

from .Util import isHandData

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


class BarGraphWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = FigureCanvas(figure.Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)

        self.nbrCategories = 0
        self.offset_nullValue = .01
        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

        self.changeCategories([])
        self.updateValues(np.random.rand(self.nbrCategories))
    
    def changeCategories(self, categories:int):
        self.clear()
        self.nbrCategories = len(categories)
        if self.nbrCategories == 0:
            bottom = 0
            top = 0
            left = 0
            right = self.offset_nullValue
            nrects = 0

        else:
            bins = np.array([float(i)/self.nbrCategories for i in range(self.nbrCategories+1)])

            bottom = bins[:-1] + (.1/self.nbrCategories)
            top = bins[1:] - (.1/self.nbrCategories)
            left = np.zeros(len(top))
            right = left + self.offset_nullValue
            nrects = len(top)

        nverts = nrects * (1 + 3 + 1)
        self.verts = np.zeros((nverts, 2))
        codes = np.full(nverts, path.Path.LINETO)
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        self.verts[0::5, 0] = left
        self.verts[0::5, 1] = bottom
        self.verts[1::5, 0] = left
        self.verts[1::5, 1] = top
        self.verts[2::5, 0] = right
        self.verts[2::5, 1] = top
        self.verts[3::5, 0] = right
        self.verts[3::5, 1] = bottom

        patch = None

        barpath = path.Path(self.verts, codes)
        patch = patches.PathPatch(
            barpath, facecolor='green', alpha=0.5) #edgecolor='yellow', 
        self.ax.add_patch(patch)

        # Add category names
        font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'fontsize': 'large',
        }

        for i, cat in enumerate(categories):
            posy = (bottom[i]*2 + top[i])/3.
            self.ax.text(0.01, posy, cat, fontdict=font)

        self.canvas.draw()

    def updateValues(self, values:np.ndarray):
        self.verts[2::5, 0] = values + self.offset_nullValue
        self.verts[3::5, 0] = values + self.offset_nullValue
        self.canvas.draw()
    
    def clear(self):
        self.ax.clear()


class HandPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = FigureCanvas(figure.Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim([-1.,1.])
        self.ax.set_ylim([-1.,1.])
        self.ax.set_aspect('equal')

        self.fingerLines = [
            lines.Line2D([], [], color='r'),
            lines.Line2D([], [], color='y'),
            lines.Line2D([], [], color='g'),
            lines.Line2D([], [], color='b'),
            lines.Line2D([], [], color='m')]

        for line in self.fingerLines:
            self.ax.add_line(line)
    
    def plotHand(self, handKeypoints):
        if isHandData(handKeypoints):
            colors = ['r','y','g','b','m']
            data = [handKeypoints[:, 0:5],
                    np.insert(handKeypoints[:, 5:9].T, 0, handKeypoints[:,0], axis=0).T,
                    np.insert(handKeypoints[:, 9:13].T, 0, handKeypoints[:,0], axis=0).T,
                    np.insert(handKeypoints[:, 13:17].T, 0, handKeypoints[:,0], axis=0).T,
                    np.insert(handKeypoints[:, 17:21].T, 0, handKeypoints[:,0], axis=0).T]
            for i,line in enumerate(self.fingerLines):
                line.set_data(data[i][0], data[i][1])
        self.canvas.draw()
    
    def clear(self):
        for line in self.fingerLines:
            line.set_data([], [])
        self.canvas.draw()


class HandAnalysisWidget(QtWidgets.QGroupBox):
    def __init__(self, handID:int, showInput:bool=True):
        super().__init__(('Right' if handID == 1 else 'left') + ' hand analysis')

        self.handID = handID
        self.showInput = showInput
        self.classOutputs = []
        self.modelClassifier = None
        self.currentPrediction = ''

        self.layout=QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        self.classGraphWidget = BarGraphWidget()

        if self.showInput:
            self.handGraphWidget = HandPlotWidget()
            self.graphSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.graphSplitter.addWidget(self.handGraphWidget)
            self.graphSplitter.addWidget(self.classGraphWidget)
            self.graphSplitter.setStretchFactor(0,2)
            self.graphSplitter.setStretchFactor(1,1)

            self.layout.addWidget(self.graphSplitter)
        else:
            self.layout.addWidget(self.classGraphWidget)
    
    def setClassifierModel(self, model, classOutputs): #model:tf.keras.models
        self.modelClassifier = model
        self.classOutputs = classOutputs
    
    def drawHand(self, handKeypoints:np.ndarray, accuracy:float):
        ''' Draw keypoints of a hand pose in the widget if showInput==True.
        
        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
            accuracy (float): Global accuracy of detection of the pose.
        '''
        if self.showInput:
            #self.handGraphWidget.setTitle('Detection accuracy: ' + str(accuracy))
            self.updatePredictedClass(handKeypoints)
            if isHandData(handKeypoints):
                self.handGraphWidget.plotHand(handKeypoints)
            else:
                self.handGraphWidget.clear()
            

    def updatePredictedClass(self, keypoints:np.ndarray):
        ''' Draw keypoints of a hand pose in the widget.
        
        Args:
            keypoints (np.ndarray((3,21),float)): Coordinates x, y and the accuracy score for each 21 key points.
        '''

        prediction = [0 for i in self.classOutputs]
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

        self.classGraphWidget.updateValues(np.array(prediction))
    
    def newModelLoaded(self, urlModel:str, classOutputs:list, handID:int):
        if TF_LOADED:
            if urlModel == 'None':
                self.modelClassifier = None
                self.classOutputs = []
                self.classGraphWidget.changeCategories(self.classOutputs)
            else:
                if handID == self.handID:
                    self.modelClassifier = tf.keras.models.load_model(urlModel)
                    self.classOutputs = classOutputs
                    self.classGraphWidget.changeCategories(self.classOutputs)
                    #self.classGraphWidget.setOpts(x=range(1,len(self.classOutputs)+1), height=[0]*len(self.classOutputs))
            
    def getCurrentPrediction(self)->str:
        return self.currentPrediction