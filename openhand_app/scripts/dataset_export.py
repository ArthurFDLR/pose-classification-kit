import json
import numpy as np
import pandas as pd
from pathlib import Path


def loadFile(filePath: Path, shuffle: bool = True):
    data_out = []
    accuracy_out = []
    with open(filePath) as f:
        data = json.load(f)
        for entry in data['data']:
            data_out.append([entry['x'], entry['y'], entry['a']])
            accuracy_out.append(entry['detection_accuracy'])

    if shuffle:
        index = np.arange(data_out.shape[0])
        np.random.shuffle(index)
        data_out = data_out[index]
        accuracy_out =  accuracy_out[index]

    return np.array(data_out), np.array(accuracy_out)

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
    data = {"label": [], "hand": [], "accuracy": []}
    for i in range(21):
        data.update({"x{}".format(i): [], "y{}".format(i): []})
    
    dataset_path = Path(".").resolve() / "Dataset" / "Hands"
    assert dataset_path.is_dir()
    exportNum = 0
    for label in labels:
        for hand in [0, 1]:
            exportNum += 1
            fileName = label + ["_left", "_right"][hand] + '_hand.json'
            filePath = dataset_path / fileName
            print(fileName)
            if filePath.is_file():
                list_data, list_accuracy = loadFile(filePath, False)
                data["label"] += [label] * list_data.shape[0]
                data["hand"] += ["left" if hand == 0 else "right"] * list_data.shape[0]
                data["accuracy"] += list(list_accuracy)

                for i in range(21):
                    data["x{}".format(i)] += list(list_data[:, 0, i])
                    data["y{}".format(i)] += list(list_data[:, 1, i])

    df = pd.DataFrame(data)
    df.to_csv(dataset_path.parent / "OpenHand_Dataset.csv", index=False)

    if len(labels)*2 == exportNum:
        print("All labels exported")
    else:
        print(len(labels)*2 - exportNum, "missing files")