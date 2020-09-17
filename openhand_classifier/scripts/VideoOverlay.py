import numpy as np
import cv2
from VideoAnalysis import VideoAnalysisThread, HandAnalysis
from Util import isHandData
import pathlib
import os

class VideoInput:
    def __init__(self, urlVideo:str):
        self.video = cv2.VideoCapture(urlVideo)
        self.frame_width = int(self.video.get(3))
        self.frame_height = int(self.video.get(4))
        self.lastRet = True

    def getFPS(self):
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = self.video.get(cv2.CAP_PROP_FPS)
        return fps       

    def getFrameNumber(self)->int:
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            frame = self.video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
        else :
            frame = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(frame)    

    def getHeight(self)->int:
        return self.frame_height

    def getWidth(self)->int:
        return self.frame_width 

    def getLastFrame(self):
        self.lastRet, frame = self.video.read()
        if self.lastRet:
            return frame
        else:
            print('Finished')
            return None
        
    def isOpened(self)->bool:
        return self.video.isOpened() and self.lastRet
    
    def release(self):
        self.video.release()

class VideoWriter:
    def __init__(self):
        self.leftHandAnalysis = HandAnalysis(0, showInput=False)
        self.rightHandAnalysis = HandAnalysis(1, showInput=False)
        self.classOutputs = []

        self.video = VideoInput('test.mp4')

        self.AnalysisThread = VideoAnalysisThread(self.video)
        self.AnalysisThread.newMat.connect(self.analyseNewImage)
        self.AnalysisThread.setResolutionStream(self.video.getWidth(), self.video.getHeight())

        self.out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), self.video.getFPS(), (1920,1080))
        
        self.AnalysisThread.start()
        self.AnalysisThread.setState(True)

        self.loadModel('24Output-2x128-17epochs')

    def analyseNewImage(self, matImage:np.ndarray): # Call each time AnalysisThread emit a new pix
        self.out.write(matImage)
        leftHandKeypoints, leftAccuracy = self.AnalysisThread.getHandData(0)
        rightHandKeypoints, rightAccuracy = self.AnalysisThread.getHandData(1)
        poseKeypoints = self.AnalysisThread.getBodyData()
        raisingLeft, raisingRight = self.AnalysisThread.isRaisingHand()

        self.leftHandAnalysis.updatePredictedClass(leftHandKeypoints)
        self.rightHandAnalysis.updatePredictedClass(rightHandKeypoints)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (255, 0, 255)

        if isHandData(leftHandKeypoints):
            position = (poseKeypoints[7][0],poseKeypoints[7][1]) # (0,0) <=> left-up corner
            cv2.putText(matImage, self.leftHandAnalysis.getCurrentPrediction(), position, font, scale, color, 2, cv2.LINE_AA)
        if isHandData(rightHandKeypoints):
            position = (poseKeypoints[4][0],poseKeypoints[4][1]) # (0,0) <=> left-up corner
            cv2.putText(matImage, self.rightHandAnalysis.getCurrentPrediction(), position, font, scale, color, 2, cv2.LINE_AA)
        
        #print(matImage)
        self.out.write(matImage)
        
    def release(self):
        self.video.release()
        self.out.release()
    
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
        
if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PyQt5.QtCore import QCoreApplication
    from PyQt5 import QtWidgets as Qtw

    import sys
    app = Qtw.QApplication(sys.argv)

    videoWriter = VideoWriter()

    while(videoWriter.video.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sys.exit(app.exec_())