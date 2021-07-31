import pandas as pd
import numpy as np

from ..config import DATASETS_PATH
from .body_models import BodyModel, BODY25, BODY18, BODY25_to_BODY18_indices


def get_one_hot(targets: np.ndarray, nb_classes: int):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def importBodyCSVDataset(testSplit: float, local_import: bool):
    """Import body dataset as numpy arrays from GitHub if available, or local dataset otherwise.

    Args:
        testSplit (float, optional): Percentage of the dataset reserved for testing. Defaults to 0.15. Must be between 0.0 and 1.0.
    """
    assert 0.0 <= testSplit <= 1.0

    datasetPath = DATASETS_PATH / "BodyPose_Dataset.csv"
    datasetURL = "https://raw.githubusercontent.com/ArthurFDLR/pose-classification-kit/master/pose_classification_kit/datasets/BodyPose_Dataset.csv"

    if local_import:
        dataset_df = pd.read_csv(datasetPath)
    else:
        dataset_df = pd.read_csv(datasetURL)

    bodyLabels_df = dataset_df.groupby("label")
    labels = list(dataset_df.label.unique())

    # Find the minimum number of samples accross categories to uniformly distributed sample sets
    total_size_cat = bodyLabels_df.size().min()
    test_size_cat = int(total_size_cat * testSplit)
    train_size_cat = total_size_cat - test_size_cat

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Iterate over each labeled group
    for label, group in bodyLabels_df:
        # remove irrelevant columns
        group_array = group.drop(["label", "accuracy"], axis=1).to_numpy()
        np.random.shuffle(group_array)

        group_array_2D = [np.array((x[::2], x[1::2])).T for x in group_array]

        x_train.append(group_array_2D[:train_size_cat])
        y_train.append([label] * train_size_cat)
        x_test.append(group_array_2D[train_size_cat : train_size_cat + test_size_cat])
        y_test.append([label] * test_size_cat)

    # Concatenate sample sets as numpy arrays
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return x_train, x_test, y_train, y_test, labels


def bodyDataset(
    testSplit: float = 0.15,
    shuffle: bool = True,
    bodyModel: BodyModel = BODY25,
    local_import: bool = False,
):
    """Return the dataset of body keypoints (see pose_classification_kit/datasets/BodyPose_Dataset.csv)
    as numpy arrays.

    Args:
        testSplit (float, optional): Percentage of the dataset reserved for testing. Defaults to 0.15. Must be between 0.0 and 1.0.
        shuffle (bool, optional): Shuffle the whole dataset. Defaults to True.
        bodyModel (BodyModel, optional): Select the keypoint format of the dataset. BODY25 or BODY18. Defaults to BODY25.
        local_import (bool, optional): Choose to use local dataset or fetch online dataset (global repository). Default False.

    Returns:
        dict: {
        'x_train': training keypoints,
        'y_train': training labels,
        'y_train_onehot': training labels one-hot encoded,
        'x_test': testing keypoints,
        'y_test': testing labels,
        'y_test_onehot': testing labels one-hot encoded,
        'labels': list of labels
    }
    """

    x_train, x_test, y_train, y_test, labels = importBodyCSVDataset(
        testSplit, local_import
    )

    # Shuffle in unison
    if shuffle:
        shuffler_train = np.random.permutation(x_train.shape[0])
        shuffler_test = np.random.permutation(x_test.shape[0])
        x_train = x_train[shuffler_train]
        x_test = x_test[shuffler_test]
        y_train = y_train[shuffler_train]
        y_test = y_test[shuffler_test]

    # Format to requested body model
    assert bodyModel in [BODY18, BODY25]
    if bodyModel == BODY18:
        x_train = x_train[:, BODY25_to_BODY18_indices]
        x_test = x_test[:, BODY25_to_BODY18_indices]

    # One-hot encoding
    y_train_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_train]), len(labels)
    )
    y_test_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_test]), len(labels)
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_onehot": y_train_onehot,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_onehot": y_test_onehot,
        "labels": labels,
    }


def handDataset(
    testSplit: float = 0.15,
    shuffle: bool = True,
    handID: int = 0,
    local_import: bool = False,
):
    """Return the dataset of hand keypoints (see pose_classification_kit/datasets/HandPose_Dataset.csv)
    as numpy arrays.

    Args:
        testSplit (float, optional): Percent of the dataset reserved for testing. Defaults to 0.15. Must be between 0.0 and 1.0.
        shuffle (bool, optional): Shuffle the whole dataset. Defaults to True.
        handID (int, optional): Select hand side - 0:left, 1:right. Default to 0.
        local_import (bool, optional): Choose to use local dataset or fetch online dataset (global repository). Default False.

    Returns:
        dict: {
        'x_train': training keypoints,
        'y_train': training labels,
        'y_train_onehot': training labels one-hot encoded,
        'x_test': testing keypoints,
        'y_test': testing labels,
        'y_test_onehot': testing labels one-hot encoded,
        'labels': list of labels
    }
    """
    assert 0.0 <= testSplit <= 1.0

    datasetPath = DATASETS_PATH / "HandPose_Dataset.csv"
    datasetURL = "https://raw.githubusercontent.com/ArthurFDLR/pose-classification-kit/master/pose_classification_kit/datasets/HandPose_Dataset.csv"

    if local_import:
        dataset_df = pd.read_csv(datasetPath)
    else:
        dataset_df = pd.read_csv(datasetURL)

    hand_label = "right" if handID else "left"
    handLabels_df = {
        hand_i: dataset_df.loc[dataset_df["hand"] == hand_i].groupby("label")
        for hand_i in ["left", "right"]
    }
    labels = list(dataset_df.label.unique())

    # Find the minimum number of samples accross categories to uniformly distributed sample sets
    total_size_cat = handLabels_df[hand_label].size().min()
    test_size_cat = int(total_size_cat * testSplit)
    train_size_cat = total_size_cat - test_size_cat

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Iterate over each labeled group
    for label, group in handLabels_df[hand_label]:
        # remove irrelevant columns
        group_array = group.drop(["label", "hand", "accuracy"], axis=1).to_numpy()
        np.random.shuffle(group_array)

        x_train.append(group_array[:train_size_cat])
        y_train.append([label] * train_size_cat)
        x_test.append(group_array[train_size_cat : train_size_cat + test_size_cat])
        y_test.append([label] * test_size_cat)

    # Concatenate sample sets as numpy arrays
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Shuffle in unison
    if shuffle:
        shuffler_test = np.random.permutation(test_size_cat * len(labels))
        shuffler_train = np.random.permutation(train_size_cat * len(labels))
        x_train = x_train[shuffler_train]
        x_test = x_test[shuffler_test]
        y_train = y_train[shuffler_train]
        y_test = y_test[shuffler_test]

    # One-hot encoding
    y_train_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_train]), len(labels)
    )
    y_test_onehot = get_one_hot(
        np.array([labels.index(sample) for sample in y_test]), len(labels)
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_onehot": y_train_onehot,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_onehot": y_test_onehot,
        "labels": labels,
    }
