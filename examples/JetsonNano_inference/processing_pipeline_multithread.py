from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/" # Specify MatplotLib config folder

print('Import dependencies...', end='\t')

import cv2, queue, threading, time, json, PIL.Image, numpy as np, collections

import torch
import torchvision.transforms as transforms
from torch2trt import TRTModule

# https://github.com/NVIDIA-AI-IOT/trt_pose
import trt_pose.coco
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

print('Done')

class VideoCapture:
    def __init__(self, stream, apiPreference):
        self.cap = cv2.VideoCapture(stream, apiPreference)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.q.put((ret, frame))
        except:
            self.cap.release()
            print('Video processing stopped')

    def read(self):
        return self.q.get()
    
    def release(self):
        return self.cap.release()

class VideoProcessing:
    def __init__(self, gstream_def:str, estimation_mode_path:str, classification_mode_path:str, topology_path:str, labels_path:str):
        self._init_poseEstimation(model_path=estimation_mode_path, topology_path=topology_path)
        self._init_poseClassification(model_path=classification_mode_path, labels_path=labels_path)
        self._init_videoCapture(gstream_def)

        buffer_size = 5
        self.poses_buffer = collections.deque(buffer_size*[str(None)], buffer_size)
        self.processing_times = collections.deque(buffer_size*[0.], buffer_size)
        
        t = threading.Thread(target=self._run)
        t.daemon = True
        t.start()

    def _init_videoCapture(self, gstream_def:str):
        print('Initialize video capture pipeline...', end='\t')

        # Normalization values can be fine-tuned for your camera. Still, default values generally perform well.
        self.video_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.video_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        self.cap = VideoCapture(gstream_def, cv2.CAP_GSTREAMER)

        print('Done')

    def _init_poseEstimation(self, model_path, topology_path):
        print('Initialize pose estimation model...', end='\t')

        with open(topology_path, 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.parse_objects = ParseObjects(topology)
        # self.draw_objects = DrawObjects(topology)

        self.model_estimation = TRTModule()
        self.model_estimation.load_state_dict(torch.load(model_path))
        
        print('Done')


    def _init_poseClassification(self, model_path, labels_path):
        print('Initialize pose classification model...', end='\t')

        model_classification = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        self.infer_classification = model_classification.signatures['serving_default']

        # Blank inference to load model
        self.infer_classification(tf.constant(
            np.random.normal(size=(1, 18, 2)).astype(np.float32),
            dtype=tf.float32,
        ))

        with open(labels_path) as f:
            self.classificationLabels = json.load(f)['labels']
        #print("labels:", self.classificationLabels)
        print('Done')

    def _run(self):
        while True:
            start_time = time.time()
            
            # Get image
            re, image = self.cap.read()
            
            if re:
                
                # TRT-Pose inference
                cmap, paf = self.get_cmap_paf(image) # Pose estimation inference
                counts, objects, peaks = self.parse_objects(cmap, paf) # Matching algorithm
                keypoints = self.get_keypoints(counts, objects, peaks) # BODY18 model formating

                # Classification inference
                label_pose = None
                keypoints = self.preprocess_keypoints(keypoints)
                if type(keypoints) != type(None):
                    x = tf.constant(np.expand_dims(keypoints, axis=0), dtype=tf.float32)
                    prediction = self.infer_classification(x)
                    label_pose = self.classificationLabels[np.argmax(prediction['dense_20'][0])]
                self.poses_buffer.appendleft(label_pose)

                # Display image locally
                # self.draw_objects(image, counts, objects, peaks)

            else:
                raise RuntimeError('Could not read image from camera')
                
            self.processing_times.appendleft(1./(time.time() - start_time))

    def preprocess(self, image):
        #global device
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.video_mean[:, None, None]).div_(self.video_std[:, None, None])
        return image[None, ...]

    def get_cmap_paf(self, image):
            data = self.preprocess(image)
            cmap, paf = self.model_estimation(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            return cmap, paf

    def get_keypoints(self, counts, objects, peaks, indexBody=0):
        #if indexBody<counts[0]:
        #    return None
        kpoint = []
        human = objects[0][indexBody]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                kpoint.append([float(peak[1]),float(peak[0])])
            else:        
                kpoint.append([None, None])
        return np.array(kpoint)

    def getLengthLimb(self, data, keypoint1: int, keypoint2: int):
        if type(data[keypoint1, 0]) != type(None) and type(data[keypoint2, 0]) != type(None):
            return np.linalg.norm([data[keypoint1, 0:2] - data[keypoint2, 0:2]])
        return 0

    def preprocess_keypoints(self, keypoints:np.ndarray):
        if type(keypoints) != type(None):
            assert keypoints.shape == (18,2)
            # Find bounding box
            min_x, max_x = float("inf"), 0.0
            min_y, max_y = float("inf"), 0.0
            for k in keypoints:
                if type(k[0]) != type(None):  # If keypoint exists
                    min_x = min(min_x, k[0])
                    max_x = max(max_x, k[0])
                    min_y = min(min_y, k[1])
                    max_y = max(max_y, k[1])

            # Centering
            np.subtract(
                keypoints[:, 0],
                (min_x + max_x) / 2.,
                where=keypoints[:, 0] != None,
                out=keypoints[:, 0],
            )
            np.subtract(
                (min_y + max_y) / 2.,
                keypoints[:, 1],
                where=keypoints[:, 0] != None,
                out=keypoints[:, 1],
            )

            # Scaling  
            normalizedPartsLength = np.array(
                [
                    self.getLengthLimb(keypoints, 6,  12) * (16.0 / 5.2),  # Torso right
                    self.getLengthLimb(keypoints, 5,  11) * (16.0 / 5.2),  # Torso left
                    self.getLengthLimb(keypoints, 0,  17) * (16.0 / 2.5),  # Neck
                    self.getLengthLimb(keypoints, 12, 14) * (16.0 / 3.6),  # Right thigh
                    self.getLengthLimb(keypoints, 14, 16) * (16.0 / 3.5),  # Right lower leg
                    self.getLengthLimb(keypoints, 11, 13) * (16.0 / 3.6),  # Left thigh
                    self.getLengthLimb(keypoints, 13, 15) * (16.0 / 3.5),  # Left lower leg
                ]
            )
            
            # Mean of non-zero lengths
            normalizedPartsLength = normalizedPartsLength[normalizedPartsLength > 0.0]
            if len(normalizedPartsLength)>0:
                scaleFactor = np.mean(normalizedPartsLength)
            else:
                return None

            # Populate None keypoints with 0s
            keypoints[keypoints == None] = 0.0

            # Normalize
            np.divide(keypoints, scaleFactor, out=keypoints[:, 0:2])

            if np.any((keypoints > 1.0) | (keypoints < -1.0)):
                return None

            return keypoints.astype('float32')
        else: return None

    def get_pose(self) -> str:
        return self.poses_buffer[0]
    
    def get_fps(self) -> float:
        return np.mean(self.processing_times)

if __name__ == "__main__":

    CAM_ASPECT_RATIO = 16./9.
    INPUT_IMG_SIZE = 224
    
    gstream_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int){capture_width:d}, height=(int){capture_height:d}, "
        "format=(string)NV12, framerate=(fraction){framerate:d}/1 ! "
        "nvvidconv top={crop_top:d} bottom={crop_bottom:d} left={crop_left:d} right={crop_right:d} flip-method={flip_method:d} ! "
        "video/x-raw, width=(int){display_width:d}, height=(int){display_height:d}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink".format(
            capture_width = int(INPUT_IMG_SIZE*CAM_ASPECT_RATIO),
            capture_height = INPUT_IMG_SIZE,
            framerate = 60,
            crop_top = 0,
            crop_bottom = INPUT_IMG_SIZE,
            crop_left = int(INPUT_IMG_SIZE*(CAM_ASPECT_RATIO-1)/2),
            crop_right = int(INPUT_IMG_SIZE*(CAM_ASPECT_RATIO+1)/2),
            flip_method = 0,
            display_width = INPUT_IMG_SIZE,
            display_height = INPUT_IMG_SIZE,
        )
    )

    video_processing = VideoProcessing(
        gstream_def=gstream_pipeline,
        estimation_mode_path='resnet18_baseline_att_224x224_trt_FP16_33MB.pth',
        classification_mode_path='./Robust_BODY18_FP32',
        topology_path='human_pose.json',
        labels_path='Robust_BODY18_Info.json',
    )

    print('\n### RUNNING ###\n')
    try:
        while True:
            time.sleep(0.1)
            print('FPS: {fps:.2f}\tLabel: {lab}'.format(fps=video_processing.get_fps(), lab=video_processing.get_pose()).ljust(80)[:80], end='\r')
    except:
        video_processing.cap.release()
        print('Video processing stopped')