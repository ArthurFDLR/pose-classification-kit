import numpy as np


def data_augmentation(x: np.ndarray,
                      y: np.ndarray,
                      augmentation_ratio: float = .5,
                      remove_specific_keypoints: list = None,
                      remove_rand_keypoints_nbr: int = None,
                      random_noise_standard_deviation: float = None,
                      scaling_factor: float = None,
                      rotation_angle: float = None,
                      scaling_factor_standard_deviation: float = None,
                      rotation_angle_standard_deviation: float = None
                      ):
    """This function adds entries in the dataset depending on different parameters.

    Args:
        x (np.ndarray): 
        y (np.ndarray): Labels
        augmentation_ratio (float, optional): [description]. Defaults to .5.
        remove_specific_keypoints (list, optional): [description]. Defaults to None.
        remove_rand_keypoints_nbr (int, optional): [description]. Defaults to None.
        random_noise_standard_deviation (float, optional): [description]. Defaults to None.
        scaling_factor (float, optional): [description]. Defaults to None.
        rotation_angle (float, optional): [description]. Defaults to None.

    Returns:
        tuple(np.ndarray, np.ndarray): [description]
    """

    size_dataset = len(x)
    number_keypoints = len(x[1])
    number_entries_to_create = size_dataset*augmentation_ratio
    new_dataset = []
    index_dataset = 0

    if type(remove_rand_keypoints_nbr) != type(None):
        list_random_keypoints = [np.random.randint(0, number_keypoints) 
                                for i in range(remove_rand_keypoints_nbr)]

    shuffler = np.random.permutation(size_dataset)
    x = x[shuffler]
    y = y[shuffler]
    
    
    while number_entries_to_create != 0:
        keypoints = []
        if type(scaling_factor_standard_deviation) != type(None):
            scaling_factor_random = np.random.normal(1, scaling_factor_standard_deviation)

        if type(rotation_angle_standard_deviation) != type(None):
            rotation_angle_random = np.random.normal(0, rotation_angle_standard_deviation)
            
        if type(remove_rand_keypoints_nbr) != type(None):
            list_random_keypoints = [np.random.randint(0, number_keypoints) 
                                    for i in range(remove_rand_keypoints_nbr)]
        for i in range(number_keypoints):
            keypoint_x = x[index_dataset][i][0]
            keypoint_y = x[index_dataset][i][1]
            if type(random_noise_standard_deviation) != type(None):
                keypoint_x += np.random.normal(0, random_noise_standard_deviation)
                keypoint_y += np.random.normal(0, random_noise_standard_deviation)
            if type(scaling_factor) != type(None):
                keypoint_x *= scaling_factor
                keypoint_y *= scaling_factor
            if type(scaling_factor_standard_deviation) != type(None):
                keypoint_x *= scaling_factor_random
                keypoint_y *= scaling_factor_random
            if type(rotation_angle) != type(None):
                theta = np.radians(rotation_angle)
                c, s = np.cos(theta), np.sin(theta)
                rotation_matrix = np.array(((c, -s), (s, c)))
                keypoint = np.array([keypoint_x,keypoint_y])
                rotated_keypoint = np.dot(rotation_matrix, keypoint)
                keypoint_x = rotated_keypoint[0]
                keypoint_y = rotated_keypoint[1]
            if type(rotation_angle_standard_deviation) != type(None):
                theta = np.radians(rotation_angle_random)
                c, s = np.cos(theta), np.sin(theta)
                rotation_matrix = np.array(((c, -s), (s, c)))
                keypoint = np.array([keypoint_x,keypoint_y])
                rotated_keypoint = np.dot(rotation_matrix, keypoint)
                keypoint_x = rotated_keypoint[0]
                keypoint_y = rotated_keypoint[1]
            if type(remove_rand_keypoints_nbr) != type(None):
                if i in list_random_keypoints:
                    keypoint_x = 0
                    keypoint_y = 0
            if type(remove_specific_keypoints) != type(None):
                if i in remove_specific_keypoints:
                    keypoint_x = 0
                    keypoint_y = 0
            # Add additionnal augmentation features
            keypoints.append([keypoint_x, keypoint_y])
        new_dataset.append(keypoints)
        index_dataset = (index_dataset + 1) % size_dataset
        number_entries_to_create -= 1
    print(len(new_dataset))
    ret = np.array(new_dataset)
    np.random.shuffle(ret)

    return (ret,y)
