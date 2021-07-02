import os
import numpy as np
import pandas as pd
from pathlib import Path
import json

def loadFile(file_path: Path, shuffle: bool = True):
    data_out = []
    accuracy_out = []
    if file_path.is_file():

        currentEntry = []

        dataFile = open(file_path)
        for i, line in enumerate(dataFile):
            if i == 0:
                info = line.split(",")
                if len(info) == 3:
                    poseName = info[0]
                    handID = int(info[1])
                    tresholdValue = float(info[2])
                else:
                    print("Not a supported dataset")
                    break
            else:
                if line[0] == "#" and line[1] != "#":  # New entry
                    currentEntry = [[], [], []]
                    accuracy_out.append(float(line[1:]))
                elif line[0] == "x":
                    listStr = line[2:].split(" ")
                    for value in listStr:
                        currentEntry[0].append(float(value))
                elif line[0] == "y":
                    listStr = line[2:].split(" ")
                    for value in listStr:
                        currentEntry[1].append(float(value))
                elif line[0] == "a":  # Last line of entry
                    listStr = line[2:].split(" ")
                    for value in listStr:
                        currentEntry[2].append(float(value))
                    data_out.append(currentEntry)

        dataFile.close()

        if shuffle:
            index = np.arange(data_out.shape[0])
            np.random.shuffle(index)
            data_out = data_out[index]

    return np.array(data_out), np.array(accuracy_out)

def writeDataToJSON(path, datasetList, accuracyList, poseName, focusID):
    """ Save the current dataset to the JSON file (URL: self.currentFilePath)."""
    fileData = {
            'info':{
                'label': poseName,
                'focus': ["left_hand", "right_hand", "body"][focusID],
                'nbr_entries': len(accuracyList),
                'threshold_value': min(accuracyList),
                'focus_id': focusID,
            },
            'data':[]
        }

    for accuracy, data in zip(accuracyList, datasetList):
        fileData['data'].append({
            'detection_accuracy': float(accuracy),
            'x': data[0].tolist(),
            'y': data[1].tolist(),
            'a': data[2].tolist()
        })

    with open(path, 'w') as outfile:
        json.dump(fileData, outfile, indent = 4)

if __name__ == "__main__":
    labels = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "Chef",
        "Help",
        "Super",
        "VIP",
        "Water",
        "Metal",
        "Dislike",
        "Loser",
        "Phone",
        "Shaka",
        "Stop",
        "Vulcan_Salute",
        "Power_Fist",
        "Horns",
        "Fight_Fist",
        "Middle_Finger",
        "Ok",
    ]

    dataset_path = Path(".").resolve() / "Dataset" / "Hands"
    assert dataset_path.is_dir()
    print(len(labels), "labels detected:")
    for label in labels:
        print(label)
        for hand in [0, 1]:
            file_path = (
                dataset_path / label / ["left_hand", "right_hand"][hand] / "data.txt"
            )
            list_data, list_accuracy = loadFile(file_path, False)
            nameFile = label + '_' + ["left_hand", "right_hand"][hand] + '.json'
            writeDataToJSON(dataset_path/nameFile, list_data, list_accuracy, label, hand)
