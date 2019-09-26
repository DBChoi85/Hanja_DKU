import os
import sys
import random
import math
import numpy as np
import skimage.io
from skimage.color import rgb2gray, convert_colorspace
import matplotlib
import matplotlib.pyplot as plt
import cv2


# Root directory of the project
#ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join("C:\\Users\\ialab\\Desktop\\maskrcnn\\Mask_RCNN\\samples\\data"))  # To find local version
#import coco
#from balloon import balloon
import balloon

# Directory to save logs and trained model
MODEL_DIR = os.path.join('C:\\Users\\ialab\\Desktop\\logs')

# Local path to trained weights file
#LOAD_DIR = 'C:\\Users\\ialab\\logs\\balloon20190911T1807\\'
LOAD_DIR = 'C:\\Users\\ialab\\logs\\load_file\\'
COCO_MODEL_PATH = os.path.join(LOAD_DIR, "mask_rcnn_balloon_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")


##################################


#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(balloon.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


#############################


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


#################################



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


##############################


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
IMAGE_DIR = 'C:\\Users\\ialab\\Desktop\\img_json\\01\\'
IMAGE_DIR_test = 'C:\\Users\\ialab\\Desktop\\img_zip\\test3\\'

file_list = os.listdir(IMAGE_DIR)

for ind in file_list:
    img_file_name = ind #sample03_1_0
    filename = os.path.join(IMAGE_DIR, img_file_name)
    '''
    try:
        image = skimage.io.imread(filename)
        results = model.detect([image], verbose=1)
    except ValueError:
        img = cv2.imread(IMAGE_DIR, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = model.detect([img], verbose=1)
    '''
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    results = model.detect([image], verbose=1)

    # Run detection

    mystat = os.stat(filename)
    mysize = mystat.st_size
    print("mystat :", mysize)

    # Visualize results
    r = results[0]
    visualize.display_instances(mysize,img_file_name, image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    #visualize2.display_instances( image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
