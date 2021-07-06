import numpy as np

class BodyModel():
    def __init__(self, mapping, pairs) -> None:
        self.mapping = mapping
        self.pairs = pairs

BODY18 = BodyModel(
    mapping = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"],
    pairs =
      [[15, 13],
       [13, 11],
       [16, 14],
       [14, 12],
       [11, 12],
       [ 5,  7],
       [ 6,  8],
       [ 7,  9],
       [ 8, 10],
       [ 1,  2],
       [ 0,  1],
       [ 0,  2],
       [ 1,  3],
       [ 2,  4],
       [ 3,  5],
       [ 4,  6],
       [17,  0],
       [17,  5],
       [17,  6],
       [17, 11],
       [17, 12]]
)

BODY25 = BodyModel(
    mapping = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe', 'right_smalltoe', 'right_heel'],
    pairs = 
      [[ 1,  8],
       [ 1,  2],
       [ 1,  5],
       [ 2,  3],
       [ 3,  4],
       [ 5,  6],
       [ 6,  7],
       [ 8,  9],
       [ 9, 10],
       [10, 11],
       [ 8, 12],
       [12, 13],
       [13, 14],
       [ 1,  0],
       [ 0, 15],
       [15, 17],
       [ 0, 16],
       [16, 18],
       [ 2, 17],
       [ 5, 18],
       [14, 19],
       [19, 20],
       [14, 21],
       [11, 22],
       [22, 23],
       [11, 24]]
)

BODY25_to_BODY18_indices = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 1]
BODY25flat_to_BODY18flat_indices = [0, 1, 16, 17, 15, 16, 18, 19, 17, 18, 5, 6, 2, 3, 6, 7, 3, 4, 7, 8, 4, 5, 12, 13, 9, 10, 13, 14, 10, 11, 14, 15, 11, 12, 1, 2]

# for obj in BODY18.mapping:
#     BODY25_to_BODY18_indices.append(BODY25.mapping.index(obj))
# for i in BODY25_to_BODY18_indices:
#     BODY25flat_to_BODY18flat_indices.append(i)
#     BODY25flat_to_BODY18flat_indices.append(i+1)

def BODY25_to_BODY18(body25_keypoints:np.ndarray):
    assert(body25_keypoints.shape == 25)
    return body25_keypoints[BODY25_to_BODY18_indices]
