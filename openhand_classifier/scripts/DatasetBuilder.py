import os
import numpy as np
import pandas as pd
from pathlib import Path

def loadFile(file_path:Path, shuffle:bool=True):
    data_out = []
    accuracy_out = []
    if file_path.is_file():

        currentEntry = []

        dataFile = open(file_path)
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
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Chef', 'Help', 'Super', 'VIP', 'Water', 'Metal', 'Dislike', 'Loser', 'Phone', 'Shaka', 'Stop', 'Spoke', 'PowerFist', 'Horns', 'FightFist', 'MiddleFinger', 'Ok']
    data = {'label':[], 'hand':[], 'accuracy':[]}
    for i in range(21):
        data.update({'x{}'.format(i) : [], 'y{}'.format(i) : []})
    
    dataset_path = Path('.').resolve() / 'Dataset'
    assert(dataset_path.is_dir())
    print(len(labels), 'labels detected:')
    for label in labels:
        print(label)
        for hand in [0,1]:
            file_path = dataset_path / label / ['left_hand', 'right_hand'][hand] / 'data.txt'
            list_data, list_accuracy = loadFile(file_path, False)
            data['label'] += [label] * list_data.shape[0]
            data['hand']  += ['left' if hand == 0 else 'right'] * list_data.shape[0]
            data['accuracy'] += list(list_accuracy)

            for i in range(21):
                data['x{}'.format(i)] += list(list_data[:,0,i])
                data['y{}'.format(i)] += list(list_data[:,1,i])
    
    df = pd.DataFrame(data)
    df.to_csv(dataset_path / 'OpenHand_dataset.csv', index=False)