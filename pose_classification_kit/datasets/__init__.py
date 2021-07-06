import pandas as pd
import numpy as np

from ..config import DATASETS_PATH
from ..src.imports.tensorflow import tf
from ..src.imports.openpose import op


def bodyDataset(testSplit:float=0.15, shuffle:bool=True):
    """ Return the dataset of body keypoints (see pose_classification_kit/datasets/BodyPose_Dataset.csv)
    as numpy arrays. 

    Args:
        testSplit (float, optional): Percent of the dataset reserved for testing. Defaults to 0.15. Must be between 0.0 and 1.0.
        shuffle (bool, optional): Shuffle the whole dataset. Defaults to True.

    Returns:
        dict: {
        'x_train': training keypoints,
        'y_train': training labels,
        'y_train_onehot': training labels one-hot encoded,
        'x_test': testing keypoints,
        'y_test': testing labels,
        'y_test_onehot': testing labels one-hot encoded,
        'labels': list of labels,
        'BODY25_Mapping': dictionnary describing the mapping of the keypoints,
        'BODY25_Pairs': Pairs of keypoints forming the skeleton,
    }
    """
    datasetPath = DATASETS_PATH / 'BodyPose_Dataset.csv'

    assert(0.<=testSplit<=1.0)
    assert(datasetPath.is_file())

    dataset_df = pd.read_csv(datasetPath)
    bodyLabels_df = dataset_df.groupby('label')
    labels = list(dataset_df.label.unique())

    # Find the minimum number of samples accross categories to uniformly distributed sample sets
    total_size_cat = bodyLabels_df.size().min()
    test_size_cat  = int(total_size_cat*testSplit)
    train_size_cat = total_size_cat - test_size_cat

    x_train = []
    x_test  = []
    y_train = []
    y_test  = []

    # Iterate over each labeled group
    for label, group in bodyLabels_df:
        # remove irrelevant columns
        group_array = group.drop(['label', 'accuracy'], axis=1).to_numpy()
        np.random.shuffle(group_array)
        
        x_train.append(group_array[:train_size_cat])
        y_train.append([label]*train_size_cat)
        x_test.append(group_array[train_size_cat : train_size_cat+test_size_cat])
        y_test.append([label]*test_size_cat)

    # Concatenate sample sets as numpy arrays and shuffle in unison
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    if shuffle:
        shuffler_test = np.random.permutation(test_size_cat*len(labels))
        shuffler_train = np.random.permutation(train_size_cat*len(labels))
        x_train = x_train[shuffler_train]
        x_test  = x_test[shuffler_test]
        y_train = y_train[shuffler_train]
        y_test  = y_test[shuffler_test]

    # One-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical([list(labels).index(sample) for sample in y_train])
    y_test_onehot  = tf.keras.utils.to_categorical([list(labels).index(sample) for sample in y_test])

    poseModel = op.PoseModel.BODY_25
    poseModelMapping = op.getPoseBodyPartMapping(poseModel)
    posePartPairs = op.getPosePartPairs(poseModel)
    posePartPairs = np.stack([posePartPairs[::2], posePartPairs[1::2]]).T

    return {
        'x_train': x_train,
        'y_train': y_train,
        'y_train_onehot': y_train_onehot,
        'x_test': x_test,
        'y_test': y_test,
        'y_test_onehot': y_test_onehot,
        'labels': labels,
        'BODY25_Mapping': poseModelMapping,
        'BODY25_Pairs': posePartPairs,
    }