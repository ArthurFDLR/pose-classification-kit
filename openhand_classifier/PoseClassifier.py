import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Avoid the annoying tf warnings 

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#plt.xkcd()
plt.style.use('ggplot')

def showModelComparison(modelsDict, sizePlot, commonDescription):
    '''Display the evolution of Loss and Accuracy versus epochs given histories of training

    Parameters:
        modelsDict (Dict{ str:Dict{ str:list[float] } }): Dictionary containing the training history of a model referenced by its name
        sizePlot (list[row,column])
        commonDescription (str): Common caracteristics of the models
    
    Return:
        Matplotlib figure: Comparison of given models
    '''

    fig, axes = plt.subplots(sizePlot[0], sizePlot[1])
    plt.text(x=0.5, y=0.94, s='Title', fontsize=20, ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.88, s= commonDescription, fontsize=17, ha="center", transform=fig.transFigure)


    for i, (nameModel, history) in enumerate(modelsDict.items()):

        try:
            ax = axes.flatten()[i]
        except:
            ax=axes

        for counter, (name, values) in enumerate(history.items()):
            ax.plot(values, label=name, marker='o', color='blue' if counter%2==0 else 'red', linestyle='--' if counter<2 else '-')
            if counter == 3:
                ax.set_title(nameModel + '\nBest accuracy ' + '): ' + str(round(max(values)*100.0,4)) + '%')
        #ax.set_xticks([i for i in range(nbrEpochsMax)])
        ax.legend()
    plt.subplots_adjust(top=0.8, wspace=0.2, right=0.95, left=0.05, bottom=0.05, hspace=0.35)
    plt.show()

def showTrainingSizeComparison(modelsDict, fileName:str, commonDescription:str, nbrEpochsMax:int=-1):
    '''Display the evolution of Loss and Accuracy versus epochs given histories of training

    Parameters:
        modelsDict (Dict{ str:Dict{ str:list[float] } }): Dictionary containing the training history of a model referenced by its name
        fileName (str): Name of the pgn file generated.
        commonDescription (str): Common caracteristics of the models.
        nbrEpochsMax (int): Epochs of training, use to set x_axis ticks. Use standard ticks if negative.
    
    Return:
        Matplotlib figure: Comparison of given models
    '''

    resultsCategories = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    nameCategories = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy']
    fig, axes = plt.subplots(2, 2)
    plt.text(x=0.5, y=0.94, s='Training dataset size comparison', fontsize=20, ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.88, s= commonDescription, fontsize=17, ha="center", transform=fig.transFigure)

    for i in range(4):
        try:
            ax = axes.flatten()[i]
        except:
            ax=axes
        
        ax.set_title(nameCategories[i])

        if nbrEpochsMax>0:
            ax.set_xticks([i+1 for i in range(nbrEpochsMax)])

        for nameModel, history in modelsDict.items():
            ax.plot([i+1 for i in range(nbrEpochsMax)], history[resultsCategories[i]], label=nameModel) #color='blue' if i%2==0 else 'red', linestyle='--' if i<2 else '-'
        ax.legend()
        #ax.grid()
    plt.subplots_adjust(top=0.8, wspace=0.2, right=0.95, left=0.05, bottom=0.05, hspace=0.35)

    fig1 = plt.gcf()
    fig1.set_size_inches((20, 12), forward=False)
    fig1.savefig('.\\ModelsComparisons\\' + fileName + '.png', dpi=300)

    axes.flatten()[0].set_ylim(0.0,0.25)
    axes.flatten()[1].set_ylim(0.95,1.02)
    axes.flatten()[2].set_ylim(0.0,0.25)
    axes.flatten()[3].set_ylim(0.95,1.02)
    fig2 = plt.gcf()
    fig2.set_size_inches((20, 12), forward=False)
    fig2.savefig('.\\ModelsComparisons\\' + fileName + '_Zoomed.png', dpi=300)

    #plt.show()

def Comparison_TrainingSize(model:tf.keras.models ,classOutput:list, handID:int, epochs:int):
    model.save_weights('.\\Data\\modelTmp.h5')
    modelsDict = {}

    for size in [25, 50, 75, 100, 125, 150, 175]:
        model.load_weights('.\\Data\\modelTmp.h5')

        trainSamples_x, trainSamples_y, testSamples_x, testSamples_y = loadDataset(classOutput, size, handID,True)

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy', # prefere loss='sparse_categorical_crossentropy' if not one-hot encoded
                    metrics=['accuracy'])

        hist = model.fit(x=trainSamples_x, y=trainSamples_y, epochs=epochs, batch_size=20,  validation_data=(testSamples_x,testSamples_y)).history

        modelsDict[str(size) + ' samples per pose'] = hist
    
    return modelsDict

def loadFile(poseName:str, handID:int, shuffle:bool=True):
    fileName = '.\\Datasets\\{}\\{}\\data.txt'.format(poseName, 'right_hand' if handID == 1 else 'left_hand')
    listOut = []
    if os.path.exists(fileName):

        currentEntry = []

        dataFile = open(fileName)
        for i, line in enumerate(dataFile):
            if i == 0:
                info = line.split(',')
                if len(info) == 3:
                    poseName = info[0]
                    handID = int(info[1])
                    tresholdValue = float(info[2])
                else:
                    print('Not a supported dataset')
                    break
            else:
                if line[0] == '#' and line[1] != '#': # New entry
                    currentEntry = [[], []]
                    accuracy = float(line[1:])
                elif line[0] == 'x':
                    listStr = line[2:].split(' ')
                    for value in listStr:
                        currentEntry[0].append(float(value))
                elif line[0] == 'y':
                    listStr = line[2:].split(' ')
                    for value in listStr:
                        currentEntry[1].append(float(value))
                    listOut.append([])
                    for i in range(len(currentEntry[0])):
                        listOut[-1].append(currentEntry[0][i])
                        listOut[-1].append(currentEntry[1][i])
                elif line[0] == 'a':  # Last line of entry
                    pass
        dataFile.close()
        
        listOut = np.array(listOut)
        if shuffle:
            index = np.arange(listOut.shape[0])
            np.random.shuffle(index)
            listOut = listOut[index]
    else:
        listOut = np.array(listOut)
    return listOut

def saveModel(model:tf.keras.models, name:str, handID:int, outputClass):
    rootPath = r'.\Models'
    if not os.path.isdir(rootPath): #Create Models directory if missing
        os.mkdir(rootPath)
    modelPath = rootPath + r".\\" + name
    if not os.path.isdir(modelPath): #Create model directory if missing
        os.mkdir(modelPath)
    
    classFile = open(modelPath+r'\class.txt',"w")
    for i,c in enumerate(outputClass):
        classFile.write((',' if i!=0 else '') + c)
    classFile.close()

    model.save( modelPath + r'\\' + name + ('_right' if handID == 1 else '_left') + '.h5')

def loadDataset(classOutput:list, samplePerClass:int, handID:int, onehotEncoding:bool=False):
    
    trainSamples_x = []
    testSamples_x = []
    trainSamples_y = []
    testSamples_y = []
    for i, className in enumerate(classOutput):
        loadedSampels = loadFile(className, handID, True).tolist()
        if samplePerClass > 0:
            trainingLoad = loadedSampels[:min(samplePerClass, len(loadedSampels)-1)]
            testingLoad = loadedSampels[min(samplePerClass, len(loadedSampels)-1):]
        else:
            trainingLoad = loadedSampels
            testingLoad = []

        trainSamples_x += trainingLoad
        testSamples_x += testingLoad
        outPut_tmp = [0]*len(classOutput)
        outPut_tmp[i] = 1
        trainSamples_y += [outPut_tmp for j in range(len(trainingLoad))] if onehotEncoding else [i for j in range(len(trainingLoad))]
        testSamples_y += [outPut_tmp for j in range(len(testingLoad))] if onehotEncoding else [i for j in range(len(testingLoad))]
    
    trainSamples_x = np.array(trainSamples_x)
    testSamples_x = np.array(testSamples_x)
    trainSamples_y = np.array(trainSamples_y)
    testSamples_y = np.array(testSamples_y)

    index = np.arange(trainSamples_x.shape[0])
    np.random.shuffle(index)
    trainSamples_x = trainSamples_x[index]
    trainSamples_y = trainSamples_y[index]
    
    index = np.arange(testSamples_x.shape[0])
    np.random.shuffle(index)
    testSamples_x = testSamples_x[index]
    testSamples_y = testSamples_y[index]
    
    if len(trainSamples_x) < samplePerClass*len(classOutput):
        print('Size expectation not met.')

    return trainSamples_x, trainSamples_y, testSamples_x, testSamples_y

if __name__ == "__main__":

    MODEL_SAVING, MODEL_ANALYSIS, DATASET_ANALYSIS = range(3)
    SELECTOR = 2

    classFingerCount = ['0', '1_Eng', '2_Eng', '2_Eu', '3_Eng', '3_Eu', '4', '5']
    classRestaurant = ['Chef', 'Help', 'Super', 'VIP', 'Water']
    classDivers = ['Metal', 'Dislike', 'Loser', 'Phone', 'Shaka', 'Stop', 'Spoke', 'PowerFist', 'Horns', 'FightFist', 'MiddleFinger']
    classOutput = classFingerCount + classRestaurant + classDivers

    if SELECTOR == MODEL_SAVING:

        for handID in [0,1]:

            allSamples_x, allSamples_y_oneHot, _, _ = loadDataset(classOutput, 200, handID,True)

            ## Model definition

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(128, input_dim=42, activation=tf.keras.activations.relu))
            model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
            model.add(tf.keras.layers.Dense(len(classOutput), activation=tf.keras.activations.softmax))

            model.summary()
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss='categorical_crossentropy', # prefere loss='sparse_categorical_crossentropy' if not one-hot encoded
                        metrics=['accuracy'])

            hist = model.fit(x=allSamples_x, y=allSamples_y_oneHot, epochs=17, batch_size=20, validation_split=0.0).history

            saveModel(model, '24Output-2x128-17epochs', handID, classOutput)
    

    if SELECTOR == MODEL_ANALYSIS:
        maxEpochsNbr = 30


        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=42, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))

        model.add(tf.keras.layers.Dense(len(classOutput), activation=tf.keras.activations.softmax))

        dictResults = Comparison_TrainingSize(model, classOutput, 0, maxEpochsNbr)
        showTrainingSizeComparison(dictResults, '2x64_Adam_Relu',
                                   '2 dense hidden layers of 64 neurons (Rectified linear activation) - Adam optimizer - Cross entropy loss - ' + str(len(classOutput)) + ' outputs categories (Softmax activation)',
                                   maxEpochsNbr)
        '''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=42, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(len(classOutput), activation=tf.keras.activations.softmax))

        dictResults = Comparison_TrainingSize(model, classOutput, 0, maxEpochsNbr)
        showTrainingSizeComparison(dictResults, '2x32_Adam_Relu',
                                   '2 dense hidden layers of 32 neurons (Rectified linear activation) - Adam optimizer - Cross entropy loss - ' + str(len(classOutput)) + ' outputs categories (Softmax activation)',
                                   maxEpochsNbr)
        

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=42, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(len(classOutput), activation=tf.keras.activations.softmax))

        dictResults = Comparison_TrainingSize(model, classOutput, 0, maxEpochsNbr)
        showTrainingSizeComparison(dictResults, '2x128_Adam_Relu',
                                   '2 dense hidden layers of 128 neurons (Rectified linear activation) - Adam optimizer - Cross entropy loss - ' + str(len(classOutput)) + ' outputs categories (Softmax activation)',
                                   maxEpochsNbr)
        '''
    
    if SELECTOR == DATASET_ANALYSIS:
        allSamples_x_left, _, _, _ = loadDataset(classOutput, -1, 0, False)
        allSamples_x_right, _, _, _ = loadDataset(classOutput, -1, 1, False)
        size = len(allSamples_x_left) + len(allSamples_x_right)
        print(str(size) + ' samples for ' + str(len(classOutput)) + ' output categories.')