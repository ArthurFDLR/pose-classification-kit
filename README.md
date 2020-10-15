# <h1 align = "center"> OpenHand classifier

[![Linting](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub](https://img.shields.io/github/license/ArthurFDLR/OpenHand-Classifier)](https://github.com/ArthurFDLR/OpenHand-Classifier/blob/master/LICENSE)

This classifier is build upon the excellent pose estimator [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose) from **CMU Perceptual Computing Lab**. A GUI has been developped to ease dataset creation and real-world testing.

  - [Installation](#installation)
  - [Under the hood](#under-the-hood)
    - [Features extraction](#features-extraction)
    - [Keypoints normalization](#keypoints-normalization)
    - [Dataset creation - *9809 samples for 24 output categories*](#dataset-creation---9809-samples-for-24-output-categories)
    - [The pose classifier - a simple ANN](#the-pose-classifier---a-simple-ann)
  - [User guide](#user-guide)

## Installation

Make sure that [`Poetry`](https://poetry.eustace.io/) is installed for Python 3.7 and above.

1. Git clone the repository - `git clone https://github.com/ArthurFDLR/OpenHand-Classifier`

2. Install the packages required - `python -m poetry install` (or use the configuration file [`.\requirements.txt`](https://github.com/ArthurFDLR/OpenHand-Classifier/blob/master/requirements.txt) in the Python 3.7+ environment of your choice)

3. You should now be able to run the application - `make run` (or `poetry run python .\openhand_classifier`)

Even if **OpenHand classifier** can run without [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose), it must be installed on your system to allow real-time hand gesture classification.

4.  Follow [OpenPose installation instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md).

5. Once the installation is completed, change the variable `OPENPOSE_PATH` ( [`.\openhand_classifier\__init__.py`](https://github.com/ArthurFDLR/OpenHand-Classifier/blob/master/openhand_classifier/__init__.py)) to the path to the OpenPose installation folder on your system.


## Under the hood

### Features extraction

The 21 hand keypoints (2D) used as input for this classifier are produced by OpenPose. The hand output format is as follow:

<img src="/.github/markdown/keypoints_hand.png" width="200">

More information can be found [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#face-and-hands). Please note that even if only hand keypoints are used, [OpenPose recquiered the whole body to be analysed](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/standalone_face_or_hand_keypoint_detector.md) in order to generate hand informations. Furtheremore keypoints coordinates are given in the frame of reference of the image feeded to OpenPose. Thus, the coordinates have to be normalized.
I addition to x, y coordinates, the accuracy of detection of each keypoints is provided. From now on, the sum of these values will be simply refered as accuracy.

### Keypoints normalization

OpenPose outputs have to be formated and normalized prior to the artificial neural network (ANN) training. Coordinates are normalized relatively to finger length and the center of gravity of the hand.

* **Scaling:** First, the length of each fingers - defined as a set of lines of the same color, see above - is calculated. The euclidian distances of all segments of a finger are sumed *- e.g.* <img src="https://render.githubusercontent.com/render/math?math=Thumb\_length = \sum_{i=0}^{3} d(\boldsymbol{k_i}, \boldsymbol{k_{i%2B1}})">.
Then, every coordinates composing the hand are divided by the greater finger length.

* **Centering:** Keypoints are centered relatively to the center of mass of the hand which in this case, is simply defined as <img src="https://render.githubusercontent.com/render/math?math=(\bar{\boldsymbol{k^x}}, \bar{\boldsymbol{k^y}})">.

<details><summary>Click to show code</summary>
<p>

```python
handKeypoints = np.array(op.Datum().handKeypoints)[handID, self.personID]

lengthFingers = [np.sqrt((handKeypoints[0,0] - handKeypoints[i,0])**2 + (handKeypoints[0,1] - handKeypoints[i,1])**2) for i in [1,5,9,13,17]] #Initialized with the length of the first segment of each fingers.
for i in range(3): #Add length of other segments for each fingers
    for j in range(len(lengthFingers)):
        lengthFingers[j] += np.sqrt((handKeypoints[1+j*4+i+1, 0] - handKeypoints[1+j*4+i, 0])**2 + (handKeypoints[1+j*4+i+1, 1] - handKeypoints[1+j*4+i, 1])**2)
normMax = max(lengthFingers)

handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]
outputArray = np.array([(handKeypoints.T[0] - handCenterX)/normMax,
                        -(handKeypoints.T[1] - handCenterY)/normMax,
                        (handKeypoints.T[2])])
```
</p>
</details>

Now that coordinates are normalized, the input data is flatten to be fed to the ANNs as a list of 42 values between -1.0 and 1.0:   <img src="https://render.githubusercontent.com/render/math?math=(k^x_0, k^y_0, k^x_1, k^y_1  \dots  k^x_{20}, k^y_{20})">

<img src="/.github/markdown/formated_hand.png" width="400">

### Dataset creation - [*9809 samples for 24 output categories*](https://github.com/ArthurFDLR/OpenHand-Classifier/tree/master/Datasets)

The dataset is composed of several classes. A class is composed of two text files, one for each hand. The dataset is structured as follow:

```
.\AppHandClassifier\Datasets
│
└───class_label_1
│   └───left_hand
│   │       data.txt
│   └───right_hand
│           data.txt
│
└───class_label_2
│   └───left_hand
│   │       data.txt
│   └───right_hand
│           data.txt
.
.
```

The first line of a *data.txt* files contains the caracteristics of the dataset: class label, hand identifier (0 for left hand, 1 for right hand) and the minimum accuracy of detection. To add comments, begin a line with *##*. A sample is (at leat) composed of 3 lines: a header giving the detection accuracy, x coordinates, y coordinates. 

<details><summary>Click to show examples - First lines of 'Super' set for right hand</summary>
<p>

```
Super,1,13.0
## Data generated the 2020-07-28 labelled Super (right hand) with a global accuracy higher than 13.0, based on OpenPose estimation.
## Data format: Coordinates x, y and accuracy of estimation a

#14.064389
x:-0.47471642 -0.38345036 -0.27814367 -0.17283674 -0.16581643 -0.07455035 0.24136995 0.26243138 0.18520646 -0.060509484 0.24136995 0.17116559 0.05883807 -0.095611796 0.22732908 0.14308357 0.030756325 -0.10965267 0.1220224 0.10798126 0.02373602
y:-0.120350584 0.12536536 0.38512218 0.6238177 0.8203904 0.13238579 0.12536536 0.097283475 0.09026304 -0.07822783 -0.043125518 -0.029084647 -0.015043774 -0.2467187 -0.19757552 -0.16247334 -0.14843246 -0.3801074 -0.36606652 -0.30990276 -0.30288246
a:0.4513099 0.52159405 0.73779285 0.7362725 0.8151489 0.8092662 0.74224406 0.4387765 0.23850155 0.797209 0.79372936 0.59578335 0.44275257 0.81076413 0.9635796 0.647649 0.5396069 0.80517197 0.8936012 0.7543843 0.52925146

#15.550782
x:-0.4933955 -0.3817585 -0.23523489 -0.109643176 -0.053824674 0.008971046 0.23224507 0.13456275 0.043857645 0.001993833 0.24619977 0.13456275 0.015948527 -0.025915554 0.22526786 0.113630846 0.001993833 -0.053824674 0.12060806 0.07874425 -0.0049836473
y:-0.113298275 0.13090765 0.36813638 0.5914105 0.779798 0.109975755 0.102998406 0.137885 0.14486235 -0.07841181 -0.06445711 -0.0225933 -0.015615954 -0.23888998 -0.19702616 -0.16213956 -0.16911678 -0.3575045 -0.350527 -0.30168596 -0.2947085
a:0.59823513 0.6402868 0.81965464 0.87657 0.9046949 0.83729064 0.8742925 0.47936943 0.43094704 0.82496655 0.87384015 0.65166384 0.5838103 0.8670102 0.9759184 0.6943432 0.5715823 0.81283325 0.8954963 0.71702033 0.62095624
```

</p>
</details>

Note that a training set of 150 samples per hand and per pose seems enough to yield good classification results. Dataset size and classification performance are study bellow for different ANN structures.
A couple of minutes of recording with the provided tool is enough to generate enough data for a pose.

### The pose classifier - a simple ANN

🚧 Under construction 🚧

## User guide

🚧 Under construction 🚧