## WORK IN PROGRESS
## CURRENTLY NOT WORKING

import numpy as np
import cv2
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

font = {
    "family": "serif",
    "weight": "normal",
    "size": "22",
    "serif": "DejaVu Sans",
}
matplotlib.rc("font", **font)

OPENPOSE_PATH = Path("C:/") / "Program files" / "OpenPose"

try:
    sys.path.append(str(OPENPOSE_PATH / "build" / "python" / "openpose" / "Release"))
    releasePATH = OPENPOSE_PATH / "build" / "x64" / "Release"
    binPATH = OPENPOSE_PATH / "build" / "bin"
    modelsPATH = OPENPOSE_PATH / "models"
    os.environ["PATH"] = (
        os.environ["PATH"] + ";" + str(releasePATH) + ";" + str(binPATH) + ";"
    )
    import pyopenpose as op

    OPENPOSE_LOADED = True
    print("OpenPose ({}) loaded.".format(str(OPENPOSE_PATH)))
except:
    OPENPOSE_LOADED = False
    print("OpenPose ({}) loading failed.".format(str(OPENPOSE_PATH)))

try:
    import tensorflow as tf

    GPU_LIST = tf.config.experimental.list_physical_devices("GPU")
    if GPU_LIST:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in GPU_LIST:
                # Prevent Tensorflow to take all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(
                    len(GPU_LIST), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    TF_LOADED = True
except:
    TF_LOADED = False


def format_data(handKeypoints, hand_id: int):
    """Return the key points of the hand seen in the image (cf. videoSource).

    Args:
        hand_id (int): 0 -> Left hand | 1 -> Right hand

    Returns:
        np.ndarray((3,21),float): Coordinates x, y and the accuracy score for each 21 key points.
                                    None if the given hand is not detected.
    """
    openhand_format = None
    personID = 0

    nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim > 2 else 0
    handAccuaracyScore = 0.0
    if nbrPersonDetected > 0:
        handAccuaracyScore = handKeypoints[hand_id, personID].T[2].sum()
        handDetected = handAccuaracyScore > 1.0
        if handDetected:
            handKeypoints = handKeypoints[hand_id, personID]
            # Initialize with the length of the first segment of each fingers
            lengthFingers = [
                np.sqrt(
                    (handKeypoints[0, 0] - handKeypoints[i, 0]) ** 2
                    + (handKeypoints[0, 1] - handKeypoints[i, 1]) ** 2
                )
                for i in [1, 5, 9, 13, 17]
            ]
            for i in range(3):  # Add length of other segments of each fingers
                for j in range(len(lengthFingers)):
                    x = (
                        handKeypoints[1 + j * 4 + i + 1, 0]
                        - handKeypoints[1 + j * 4 + i, 0]
                    )
                    y = (
                        handKeypoints[1 + j * 4 + i + 1, 1]
                        - handKeypoints[1 + j * 4 + i, 1]
                    )
                    lengthFingers[j] += np.sqrt(x ** 2 + y ** 2)
            normMax = max(lengthFingers)

            handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
            handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]

            outputArray = np.array(
                [
                    (handKeypoints.T[0] - handCenterX) / normMax,
                    -(handKeypoints.T[1] - handCenterY) / normMax,
                    (handKeypoints.T[2]),
                ]
            )

            openhand_format = []
            for i in range(outputArray.shape[1]):
                openhand_format.append(outputArray[0, i])  # add x
                openhand_format.append(outputArray[1, i])  # add y
            openhand_format = np.array(openhand_format)

    return openhand_format, handAccuaracyScore


def getFPS(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def getFrameNumber(video) -> int:
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        frame = video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
    else:
        frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(frame)


def getHeight(video) -> int:
    return int(video.get(4))


def getWidth(video) -> int:
    return int(video.get(3))


def create_plot(classifier_labels, prediction_probabilities, save_url):
    assert len(classifier_labels) == len(prediction_probabilities)
    fig, ax = plt.subplots(figsize=(4, 10))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.04)
    plt.box(on=None)
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.tick_params(
        axis="y", direction="in", pad=-50, which="both", left=False, labelleft=True
    )
    ax.set_yticks(np.arange(len(prediction_probabilities)))
    ax.set_yticklabels(classifier_labels, ha="left")
    ax.barh(
        np.arange(len(prediction_probabilities)),
        prediction_probabilities,
        color="#9500ff",
    )
    fig.savefig(save_url, transparent=True, dpi=108, pad_inches=0.0)
    plt.close(fig)

def run():
    current_path = Path.cwd()

    # Load Keras model
    classifier_name = "24Output-2x128-17epochs"
    classifier_path = current_path / "Models" / classifier_name
    right_hand_classifier = tf.keras.models.load_model(
        classifier_path / (classifier_name + "_right.h5")
    )
    left_hand_classifier = tf.keras.models.load_model(
        classifier_path / (classifier_name + "_left.h5")
    )
    hand_classifiers = (left_hand_classifier, right_hand_classifier)

    if os.path.isfile(classifier_path / "class.txt"):
        with open(classifier_path / "class.txt", "r") as file:
            first_line = file.readline()
            classifier_labels = first_line.split(",")
    for i in range(len(classifier_labels)):
        classifier_labels[i] = classifier_labels[i].replace("_", " ")

    # Open video
    video_path = current_path / "video" / "hand_gesture_cc.mp4"
    video_in = cv2.VideoCapture(str(video_path))
    video_nbr_frame = getFrameNumber(video_in)

    # Create output video
    outputs_name = "output"
    video_out_path = current_path / "video" / "output" / (outputs_name + ".avi")
    barchart_out_path = current_path / "video" / "output" / (outputs_name + "_barchart")
    barchart_out_path.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_out = cv2.VideoWriter(
        str(video_out_path),
        fourcc,
        getFPS(video_in),
        (getWidth(video_in), getHeight(video_in)),
    )

    # Load OpenPose
    params = dict()
    params["model_folder"] = str(modelsPATH)
    params["face"] = True
    params["hand"] = True
    params["disable_multi_thread"] = False
    netRes = 22  # Default 22
    params["net_resolution"] = "-1x" + str(16 * netRes)

    opWrapper = op.WrapperPython()
    datum = op.Datum()
    opWrapper.configure(params)
    opWrapper.start()

    # Analyse video
    print("\n\nPress 'q' to stop analysis")
    for frame_index in range(video_nbr_frame):
        if not video_in.isOpened():
            break
        else:
            hand_id = 1

            # Get frame
            ret, frame = video_in.read()

            # OpenPose analysis
            if type(frame) != type(None):
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])
                frame = datum.cvOutputData
            else:
                break
            wrists_positions = [(0, 0), (0, 0)]
            if datum.poseKeypoints.ndim > 1:
                body_keypoints = np.array(datum.poseKeypoints[0])
                wrists_positions = [
                    (body_keypoints[7][0], body_keypoints[7][1]),
                    (body_keypoints[4][0], body_keypoints[4][1]),
                ]
            hand_keypoints = np.array(datum.handKeypoints)
            hand_data, _ = format_data(hand_keypoints, hand_id)

            # OpenHand analysis
            prediction_label = ""
            prediction_probabilities = np.zeros(len(classifier_labels))
            if type(hand_data) != type(None):
                prediction_probabilities = hand_classifiers[hand_id].predict(
                    np.array([hand_data])
                )[0]
                prediction_label = classifier_labels[
                    np.argmax(prediction_probabilities)
                ]
            prediction_label = prediction_label.replace("_", " ")

            # Overlay result on video
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2
            thickness = 2
            color = (255, 0, 149)
            (label_width, label_height), baseline = cv2.getTextSize(
                prediction_label, font, scale, thickness
            )
            txt_position = tuple(
                map(
                    lambda i, j: int(i - j),
                    wrists_positions[hand_id],
                    (label_width + 80, 70),
                )
            )
            cv2.putText(
                frame,
                prediction_label,
                txt_position,
                font,
                scale,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

            # Display image
            cv2.imshow("frame", frame)

            # Write image
            video_out.write(frame)

            # Create probabilities barchart
            create_plot(
                classifier_labels[:-1],
                prediction_probabilities[:-1],
                barchart_out_path / "{}.png".format(frame_index),
            )

            # Control
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" and OPENPOSE_LOADED:
    run()