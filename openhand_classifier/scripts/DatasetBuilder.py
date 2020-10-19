import os
import numpy as np
import pandas as pd

def loadFile(poseName:str, handID:int, shuffle:bool=True):
    fileName = '.\\Datasets\\{}\\{}\\data.txt'.format(poseName, 'right_hand' if handID == 1 else 'left_hand')
    data_out = []
    accuracy_out = []
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
                    accuracy_out.append(float(line[1:]))
                elif line[0] == 'x':
                    listStr = line[2:].split(' ')
                    for value in listStr:
                        currentEntry[0].append(float(value))
                elif line[0] == 'y':
                    listStr = line[2:].split(' ')
                    for value in listStr:
                        currentEntry[1].append(float(value))
                elif line[0] == 'a':  # Last line of entry
                    data_out.append(currentEntry)

        dataFile.close()

        if shuffle:
            index = np.arange(data_out.shape[0])
            np.random.shuffle(index)
            data_out = data_out[index]

    return np.array(data_out), np.array(accuracy_out)

if __name__ == "__main__":
    labels = ['0', '1_Eng', '2_Eng', '2_Eu', '3_Eng', '3_Eu', '4', '5', 'Chef', 'Help', 'Super', 'VIP', 'Water', 'Metal', 'Dislike', 'Loser', 'Phone', 'Shaka', 'Stop', 'Spoke', 'PowerFist', 'Horns', 'FightFist', 'MiddleFinger']

    data = {'label':[], 'hand':[], 'accuracy':[]}
    for i in range(21):
        data.update({'x{}'.format(i) : [], 'y{}'.format(i) : []})
    
    for label in labels:
        for hand in [0,1]:
            list_data, list_accuracy = loadFile(label, hand, False)

            data['label'] += [label] * list_data.shape[0]
            data['hand']  += ['left' if hand == 0 else 'right'] * list_data.shape[0]
            data['accuracy'] += list(list_accuracy)

            for i in range(21):
                data['x{}'.format(i)] += list(list_data[:,0,i])
                data['y{}'.format(i)] += list(list_data[:,1,i])
    
    df = pd.DataFrame(data)
    df.to_csv('./Datasets/dataset.csv', index=False)