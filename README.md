<h1 align = "center"> Gesture Controlled Drone </h1>

[![PyPI][PyPI-shield]][PyPI-url]
[![PyV][PyV-shield]][PyV-url]
[![lint][lint-shield]][lint-url]
[![linkedin][linkedin-shield]][linkedin-url]

<p align="center">
    <img src="https://github.com/ArthurFDLR/pose-classification-kit/blob/master/.github/markdown/pck-app.PNG?raw=true" alt="Banner" width="100%" style="border-radius: 5px;">
</p>

This Python package focus on the deployment of gesture control systems. It ease dataset creation, models evaluation, and processing pipeline deployment. The critical element in the proposed processing architecture is the intermediate representation of human bodies as key points to perform efficient classification. In addition to the main application, the package contains two datasets for body/hands pose classificaiton, several classification models, and data augmentation tools that can be accessed through an API. Feel free to check-out the [**drone-gesture-control repository**](https://github.com/ArthurFDLR/drone-gesture-control) for a deployment example on Jetson Nano using this package.


- [Getting Started](#getting-started)
  - [Step 1 - Install the package](#step-1---install-the-package)
    - [Using PyPi](#using-pypi)
    - [From source](#from-source)
  - [Step 2 - Install OpenPose](#step-2---install-openpose)
  - [Step 3 - Launch application](#step-3---launch-application)
  - [Step 4 - Create new classification models](#step-4---create-new-classification-models)
- [Demonstrations](#demonstrations)
- [User guide](#user-guide)
  - [Real-time pose classification](#real-time-pose-classification)
  - [Create and manipulate datasets](#create-and-manipulate-datasets)
  - [Additional scripts](#additional-scripts)
- [Documentation](#documentation)
  - [Body datasets](#body-datasets)
  - [Data augmentation](#data-augmentation)
- [License](#license)

## Getting Started

### Step 1 - Install the package

#### Using PyPi

Run the following command to install the whole package in the desired Python environment:
  
  ```
  pip install pose-classification-kit[app]
  ```

If you don't plan to use the application but just want access to the datasets and pre-trained models:

  ```
  pip install pose-classification-kit
  ```

#### From source

Ensure that [`Poetry`](https://poetry.eustace.io/) is installed for Python 3.7 and above on your system.

1. Git clone the repository
   
    ```
    git clone https://github.com/ArthurFDLR/pose-classification-kit.git
    cd pose-classification-kit
    ```

2. Create an adequate `venv` virtual environment
   
    ```
    python -m poetry install
    ```

### Step 2 - Install OpenPose

The dataset creation and real-time model evaluation application heavily rely on the pose estimation system [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose). It must be installed on your system to allow real-time gesture classification. This step is not requiered if you don't plan to use the application.

1.  Follow [OpenPose installation instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/doc/installation).

2. Once the installation is completed, change the variable `OPENPOSE_PATH` ( [`.\pose-classification-kit\config.py`](https://github.com/ArthurFDLR/pose-classification-kit/blob/master/pose_classification_kit/config.py)) to the location of the OpenPose installation folder on your system.

### Step 3 - Launch application

You should now be able to run the application if you installed all optionnal dependancies. See the usage section about how to use the app.
```
pose-classification-app
```

### Step 4 - Create new classification models

The [`.\examples`](https://github.com/ArthurFDLR/pose-classification-kit/blob/master/examples) folder contains Jupyter Notebook detailing the use of the API to create new classification models. Note that these Notebooks can be executed on Google Colab.

## Demonstrations

<a href="https://youtu.be/FK-1G749cIo"><p align="center">
    <img src="https://github.com/ArthurFDLR/pose-classification-kit/blob/master/.github/markdown/video_embed_1.PNG?raw=true" alt="Demonstration video 1" width="70%" style="border-radius: 5px;">
</p></a>

<a href="https://youtu.be/FZAUPmKiSXg"><p align="center">
    <img src="https://github.com/ArthurFDLR/pose-classification-kit/blob/master/.github/markdown/video_embed_2.PNG?raw=true" alt="Demonstration video 2" width="70%" style="border-radius: 5px;">
</p></a>

## User guide

### Real-time pose classification

The video stream of the selected camera is fed to OpenPose at all times. The analysis results are displayed on the left side of the application. You have to choose one of the available models in the drop-down at the bottom of the analysis pannel. Keypoints extracted from the video by OpenPose are automatically normalized and fed to the classifier.

### Create and manipulate datasets

First, you either have to load or create a new set of samples for a specific label and hand side. To do so, respectively choose *Open (Ctrl+O)* or *Create new (Ctrl+N)* in *Dataset* of the menu bar. You have to specify the hand side, the label, and the newly created samples set' accuracy threshold. A configuration window will ask for the label and the newly created samples set's accuracy threshold in case of creating a new class. The accuracy threshold defines the minimum accuracy of hand keypoints detection from OpenPose of any sample in the set. This accuracy is displayed on top of the keypoints graph.

Now that a set is loaded in the application, you can record new samples from your video feed or inspect the set and delete inadequate samples. When your done, save the set through *Dataset -> Save (Ctrl+S)*.

### Additional scripts

Some functionalities  are currently unavailable through the GUI:
- You can export all dataset samples from [`.\pose_classification_kit\datasets\Body`](https://github.com/ArthurFDLR/pose-classification-kit/tree/master/pose_classification_kit/datasets/Body) and [`.\pose_classification_kit\datasets\Hands`](https://github.com/ArthurFDLR/pose-classification-kit/tree/master/pose_classification_kit/datasets/Hands) in two respective CSV files. 
  ```
  export-datasets
  ```
- You can generate videos similar to [this one](https://youtu.be/FK-1G749cIo) ([`.\pose-classification-kit\scripts\video_creation.py`](https://github.com/ArthurFDLR/OpenHand-App/tree/master/pose-classification-kit/scripts/video_creation.py) might need some modification to fit your use case).
  
  🚧 Currently not functional 🚧
  
  ```
  video-overlay
  ```

## Documentation

### Body datasets

There is a total of 20 body dataset classes which contains between 500 and $600$ samples each for a total of 10680 entries. Even if the number of samples from one class to the other varies in the raw dataset, the API yields a balanced dataset of 503 samples per class. Also, by default, 20% of these are reserved for final testing of the model. Each entry in the dataset is an array of 25 2D coordinates. The mapping of these keypoints follows the BODY25 body model. We created the dataset using the BODY25 representation as it is one of the most comprehensive standard body models. However, some pose estimation models, such as the one used on the Jetson Nano, use an 18 keypoints representation (BODY18). The seven missing keypoints do not strongly influence classification as 6 of them are used for feet representation, and the last one is a central hip keypoint. Still, the dataset must be converted to the BODY18 representation. This is done by reindexing the samples based on the comparison of the mapping of both body models. You can choose which body model to use when importing the dataset with the API.

<p align="center">
    <img src="https://github.com/ArthurFDLR/pose-classification-kit/blob/master/.github/markdown/class_body.png?raw=true" alt="Full body classes" width="80%" style="border-radius: 5px;">
</p>

<p align="center">
    <img src="https://github.com/ArthurFDLR/pose-classification-kit/blob/master/.github/markdown/body_models.png?raw=true" alt="Body models" width="80%" style="border-radius: 5px;">
</p>

### Data augmentation

The data augmentation tool currently support the following operations:

- **Scaling**: a random scaling factor drawn from a normal distribution of mean 0 and standard deviation σₛ is applied to all sample coordinates.
- **Rotation**: a rotation of an angle randomly drawn from a normal distribution of mean 0 and standard deviation σᵣ is applied to the sample.
- **Noise**: Gaussian noise of standard deviation σₙ is added to coordinates of the sample.
- **Remove keypoints**: a pre-defined or random list of keypoints are removed (coordinates set to 0) from the sample.

<details><summary>See example</summary>
<p>


<table>
    <tr>
        <td>Augmentation Ratio</td>
        <td>σₛ</td>
        <td>σᵣ</td>
        <td>σₙ</td>
        <td>Remove keypoints</td>
    </tr>
    <tr>
        <td>10%</td>
        <td>0.08</td>
        <td>0.0</td>
        <td>0.0</td>
        <td>None</td>
    </tr>
    <tr>
        <td>10%</td>
        <td>0.0</td>
        <td>10.0</td>
        <td>0.0</td>
        <td>None</td>
    </tr>
    <tr>
        <td>15%</td>
        <td>0.0</td>
        <td>0.0</td>
        <td>0.03</td>
        <td>Legs</td>
    </tr>
    <tr>
        <td>15%</td>
        <td>0.0</td>
        <td>0.0</td>
        <td>0.03</td>
        <td>Legs & Hip</td>
    </tr>
    <tr>
        <td>20%</td>
        <td>0.0</td>
        <td>0.0</td>
        <td>0.03</td>
        <td>2 random</td>
    </tr>
</table>

```python
from pose_classification_kit.datasets import BODY18, bodyDataset, dataAugmentation

dataset = bodyDataset(testSplit=.2, shuffle=True, bodyModel=BODY18)
x_train = dataset['x_train']
y_train = dataset['y_train_onehot']
x, y = [x_train], [y_train]

# Scaling augmentation
x[len(x):],y[len(y):] = tuple(zip(dataAugmentation(
    x_train, y_train,
    augmentation_ratio=.1,
    scaling_factor_standard_deviation=.08,
)))

# Rotation augmentation
x[len(x):],y[len(y):] = tuple(zip(dataAugmentation(
    x_train, y_train,
    augmentation_ratio=.1,
    rotation_angle_standard_deviation=10,
)))

# Upper-body augmentation
lowerBody_keypoints = np.where(np.isin(BODY18.mapping,[
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]))[0]
x[len(x):],y[len(y):] = tuple(zip(dataAugmentation(
    x_train, y_train,
    augmentation_ratio=.15,
    remove_specific_keypoints=lowerBody_keypoints,
    random_noise_standard_deviation=.03
)))
lowerBody_keypoints = np.where(np.isin(BODY18.mapping,[
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_hip", "right_hip",
]))[0]
x[len(x):],y[len(y):] = tuple(zip(dataAugmentation(
    x_train, y_train,
    augmentation_ratio=.15,
    remove_specific_keypoints=lowerBody_keypoints,
    random_noise_standard_deviation=.03
)))        

# Random partial input augmentation
x[len(x):],y[len(y):] = tuple(zip(dataAugmentation(
    x_train, y_train,
    augmentation_ratio=.2,
    remove_rand_keypoints_nbr=2,
    random_noise_standard_deviation=.03
)))

x_train_augmented = np.concatenate(x, axis=0)
y_train_augmented = np.concatenate(y, axis=0)
```
</p>
</details>


<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/ArthurFDLR/pose-classification-kit/blob/main/LICENSE) for more information.


<!-- MARKDOWN LINKS & IMAGES -->
[PyPI-shield]: https://img.shields.io/pypi/v/pose-classification-kit?style=for-the-badge
[PyPI-url]: https://pypi.org/project/pose-classification-kit/

[PyV-shield]: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=for-the-badge
[PyV-url]: https://github.com/ArthurFDLR/pose-classification-kit/blob/master/pyproject.toml

[lint-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[lint-url]: https://github.com/psf/black

[license-shield]: https://img.shields.io/github/license/ArthurFDLR/OpenHand-Classifier?style=for-the-badge
[license-url]: https://github.com/ArthurFDLR/OpenHand-Classifier/blob/master/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/arthurfdlr/

<!--
## Under the hood

### Features extraction

The 21 hand keypoints (2D) used as input for this classifier are produced by OpenPose. The hand output format is as follow:

<img src="https://raw.githubusercontent.com/ArthurFDLR/pose-classification-kit/master/.github/markdown/keypoints_hand.png" width="200">

More information are available [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md). Please note that even though OpenHand focuses on hand keypoints, OpenPose requires the whole body to be analyzed to generate hand data. Furthermore, keypoints coordinates are given in the frame of reference of the image fed to OpenPose. Thus, the coordinates have to be normalized.
I addition to x, y coordinates, the accuracy of detection of each key points is provided.

### Keypoints normalization

OpenPose outputs have to be formatted and normalized before classification analysis. Coordinates are normalized relative to finger length and the center of gravity of the hand.

* **Scaling:** First, the length of each fingers - defined as a set of lines of the same color, see above - is calculated. The euclidian distances of all segments of a finger are sumed *- e.g.* <img src="https://render.githubusercontent.com/render/math?math=Thumb\_length = \sum_{i=0}^{3} d(\boldsymbol{k_i}, \boldsymbol{k_{i%2B1}})">.
Then, every coordinates composing the hand are divided by the greater finger length.

* **Centering:** Keypoints are centered relative to the center of mass of the hand which, in this case, is simply defined as <img src="https://render.githubusercontent.com/render/math?math=(\bar{\boldsymbol{k^x}}, \bar{\boldsymbol{k^y}})">.

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

<img src="https://raw.githubusercontent.com/ArthurFDLR/pose-classification-kit/master/.github/markdown/formated_hand.png" width="400">

### Dataset creation - [*11090 samples for 27 categories*](https://github.com/ArthurFDLR/OpenHand-Classifier/tree/master/Datasets)

The dataset is composed of several classes consisting of two text files, one for each hand. The dataset is structured as follow:

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

The first line of a *data.txt* file contains the set's characteristics:
- Class label
- Hand identifier (0 for the left hand, 1 for the right hand)
- The minimum accuracy of detection

To add comments, begin a line with *##*. A sample is (at least) composed of 3 lines: a header giving the detection accuracy, x coordinates, y coordinates. 

<details><summary>Click to show examples - First lines of 'Super' set for right hand</summary>
<p>

```
Super,1,13.0
## Data generated the 2020-07-28 labeled Super (right hand) with a global accuracy higher than 13.0, based on OpenPose estimation.
## Data format: Coordinates x, y, and accuracy of estimation a

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

Note that a training set of 150 samples per hand and per pose seems enough to yield good classification results. A couple of minutes of recording with the provided tool is enough to generate enough data for a pose.

### Pose classifier models

Classification models available in the application are stored in [`.\Models`](https://github.com/ArthurFDLR/OpenHand-App/tree/master/Models). Each model sub-folder contains two HDF5 files containing the model's architecture and weights values. While both models usually share the same architecture, they are trained to analyze the right or the left hand. Besides, a text file `class.txt` provides labels associated with the classifiers' one-hot encoded output.

```
.\AppHandClassifier\Models
│
└───model_1
|       class.txt
│       model_1_left.h5
|       model_1_right.h5
│
└───model_2
|       class.txt
│       model_2_left.h5
|       model_2_right.h5
.
.
```

See [**OpenHand-Models** repository](https://github.com/ArthurFDLR/OpenHand-Models) for more details about model creation.

## User guide

### Real-time pose classification

The video feed of the selected camera is fed to OpenPose at all times. The analysis results are displayed on the left side of the application. You have to choose one of the available models in the drop-down at the bottom of the hand-analysis window. Hand keypoints extracted from the video feed by OpenPose are automatically normalized and fed to the classifier.

### Create and manipulate datasets

First, you either have to load or create a new set of samples for a specific label and hand side. To do so, respectively choose *Open (Ctrl+O)* or *Create new (Ctrl+N)* in *Dataset* of the menu bar. You have to specify the hand side, the label, and the newly created samples set' accuracy threshold. The accuracy threshold defines the minimum accuracy of hand keypoints detection from OpenPose of any sample in the set. This accuracy is displayed on top of hand keypoints graphs.

Now that a set is loaded in the application, you can record new samples from your video feed or inspect the set and delete inadequate samples. When your done, save the set through *Dataset -> Save (Ctrl+S)*.

### Additional scripts

Some functionalities  are currently unavailable through the GUI:
- You can export all dataset samples from [`.\Dataset`](https://github.com/ArthurFDLR/OpenHand-App/tree/master/Dataset) in a single CSV file - `make dataset-csv` (or `poetry run python .\pose-classification-kit\scripts\dataset_export.py`)
- You can generate videos similar to [this one](https://youtu.be/FK-1G749cIo) ([`.\pose-classification-kit\scripts\video_creation.py`](https://github.com/ArthurFDLR/OpenHand-App/tree/master/pose-classification-kit/scripts/video_creation.py) might need some modification to fit your use case) - `make video-overlay` (or `poetry run python .\pose-classification-kit\scripts\video_creation.py`)
-->