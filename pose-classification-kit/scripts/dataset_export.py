import json
import numpy as np
import pandas as pd
from pathlib import Path


def loadFile(filePath: Path, shuffle: bool = True):
    data_out = []
    accuracy_out = []
    with open(filePath) as f:
        data = json.load(f)
        for entry in data["data"]:
            data_out.append([entry["x"], entry["y"], entry["a"]])
            accuracy_out.append(entry["detection_accuracy"])

    if shuffle:
        index = np.arange(data_out.shape[0])
        np.random.shuffle(index)
        data_out = data_out[index]
        accuracy_out = accuracy_out[index]

    return np.array(data_out), np.array(accuracy_out)


def generateHandDataset(labels, dataset_path):
    data = {"label": [], "hand": [], "accuracy": []}
    for i in range(21):
        data.update({"x{}".format(i): [], "y{}".format(i): []})

    for label in labels:
        for hand in [0, 1]:
            fileName = label + ["_left", "_right"][hand] + "_hand.json"
            filePath = dataset_path / fileName

            if filePath.is_file():
                list_data, list_accuracy = loadFile(filePath, False)
                data["label"] += [label] * list_data.shape[0]
                data["hand"] += ["left" if hand == 0 else "right"] * list_data.shape[0]
                data["accuracy"] += list(list_accuracy)

                for i in range(21):
                    data["x{}".format(i)] += list(list_data[:, 0, i])
                    data["y{}".format(i)] += list(list_data[:, 1, i])

                print(fileName, "imported")
            else:
                print(fileName, "not found")
    return data


def generateBodyDataset(labels, dataset_path):
    data = {"label": [], "accuracy": []}
    for i in range(25):
        data.update({"x{}".format(i): [], "y{}".format(i): []})

    for label in labels:
        fileName = label + "_body.json"
        filePath = dataset_path / fileName

        if filePath.is_file():
            list_data, list_accuracy = loadFile(filePath, False)
            data["label"] += [label] * list_data.shape[0]
            data["accuracy"] += list(list_accuracy)

            for i in range(25):
                data["x{}".format(i)] += list(list_data[:, 0, i])
                data["y{}".format(i)] += list(list_data[:, 1, i])

            print(fileName, "imported")
        else:
            print(fileName, "not found")
    return data


def run():
    handLabels = [
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

    bodyLabels = [
        "Seated",
        "Stand_RightArmRaised",
        "standing",
        "T",
        "Tree_left",
        "Tree_right",
        "UpwardSalute",
        "Warrior2_left",
        "Warrior2_right",
    ]

    dataset_path = Path(".").resolve() / "Dataset"

    datasetHands = pd.DataFrame(generateBodyDataset(bodyLabels, dataset_path / "Body"))
    datasetHands.to_csv(dataset_path / "BodyPose_Dataset.csv", index=False)

    datasetBody = pd.DataFrame(generateHandDataset(handLabels, dataset_path / "Hands"))
    datasetBody.to_csv(dataset_path / "HandPose_Dataset.csv", index=False)


if __name__ == "__main__":
    run()
