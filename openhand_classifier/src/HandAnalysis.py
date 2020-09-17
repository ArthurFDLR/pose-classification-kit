import pyqtgraph as pg
import numpy as np
import os

from .qt import QtWidgets, QtCore

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
        if TF_LOADED:
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