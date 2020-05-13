#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from xml.etree import ElementTree
from xml.dom import minidom
import collections
import matplotlib as matplot
import seaborn as sns
import cv2
import shutil
from absl import app
from absl import flags
from absl import logging
import glob
import alignment
from alignment import compute_overlap
from alignment import align


# In[32]:


#video_path
base_path = "/home/ubuntu/data/raw_data_KITTI/"
# Root directory of the RCNN project
ROOT_DIR = os.path.abspath("../Mask_RCNN")
# result WIDTH and HEIGHT
WIDTH = 416
HEIGHT = 128
# calib_cam_to_cam.txt path
INPUT_TXT_FILE="./calib_cam_to_cam.txt"
# result seq length
SEQ_LENGTH = 3
# result step size
STEPSIZE = 1
#result output dir
OUTPUT_DIR = "/home/ubuntu/data/kitti_result_all_20200513"
#temp data dir
TEMP_DIR="/home/ubuntu/data/train_data_example_all_20200513/"


# In[33]:


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

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


# In[34]:


test_dirs=["2011_09_26_drive_0117",
"2011_09_28_drive_0002",
"2011_09_26_drive_0052",
"2011_09_30_drive_0016",
"2011_09_26_drive_0059",
"2011_09_26_drive_0027",
"2011_09_26_drive_0020",
"2011_09_26_drive_0009",
"2011_09_26_drive_0013",
"2011_09_26_drive_0101",
"2011_09_26_drive_0046",
"2011_09_26_drive_0029",
"2011_09_26_drive_0064",
"2011_09_26_drive_0048",
"2011_10_03_drive_0027",
"2011_09_26_drive_0002",
"2011_09_26_drive_0036",
"2011_09_29_drive_0071",
"2011_10_03_drive_0047",
"2011_09_30_drive_0027",
"2011_09_26_drive_0086",
"2011_09_26_drive_0084",
"2011_09_26_drive_0096",
"2011_09_30_drive_0018",
"2011_09_26_drive_0106",
"2011_09_26_drive_0056",
"2011_09_26_drive_0023",
"2011_09_26_drive_0093"]

data_dirs=["2011_09_26_drive_0001",
"2011_09_26_drive_0002",
"2011_09_26_drive_0005",
"2011_09_26_drive_0009",
"2011_09_26_drive_0011",
"2011_09_26_drive_0013",
"2011_09_26_drive_0014",
"2011_09_26_drive_0015",
"2011_09_26_drive_0017",
"2011_09_26_drive_0018",
"2011_09_26_drive_0019",
"2011_09_26_drive_0020",
"2011_09_26_drive_0022",
"2011_09_26_drive_0023",
"2011_09_26_drive_0027",
"2011_09_26_drive_0028",
"2011_09_26_drive_0029",
"2011_09_26_drive_0032",
"2011_09_26_drive_0035",
"2011_09_26_drive_0036",
"2011_09_26_drive_0039",
"2011_09_26_drive_0046",
"2011_09_26_drive_0048",
"2011_09_26_drive_0051",
"2011_09_26_drive_0052",
"2011_09_26_drive_0056",
"2011_09_26_drive_0057",
"2011_09_26_drive_0059",
"2011_09_26_drive_0060",
"2011_09_26_drive_0061",
"2011_09_26_drive_0064",
"2011_09_26_drive_0070",
"2011_09_26_drive_0079",
"2011_09_26_drive_0084",
"2011_09_26_drive_0086",
"2011_09_26_drive_0087",
"2011_09_26_drive_0091",
"2011_09_26_drive_0093",
"2011_09_26_drive_0095",
"2011_09_26_drive_0096",
"2011_09_26_drive_0101",
"2011_09_26_drive_0104",
"2011_09_26_drive_0106",
"2011_09_26_drive_0113",
"2011_09_26_drive_0117",
"2011_09_26_drive_0119",
"2011_09_28_drive_0001",
"2011_09_28_drive_0002",
"2011_09_28_drive_0016",
"2011_09_28_drive_0021",
"2011_09_28_drive_0034",
"2011_09_28_drive_0035",
"2011_09_28_drive_0037",
"2011_09_28_drive_0038",
"2011_09_28_drive_0039",
"2011_09_28_drive_0043",
"2011_09_28_drive_0045",
"2011_09_28_drive_0047",
"2011_09_28_drive_0053",
"2011_09_28_drive_0054",
"2011_09_28_drive_0057",
"2011_09_28_drive_0065",
"2011_09_28_drive_0066",
"2011_09_28_drive_0068",
"2011_09_28_drive_0070",
"2011_09_28_drive_0071",
"2011_09_28_drive_0075",
"2011_09_28_drive_0077",
"2011_09_28_drive_0078",
"2011_09_28_drive_0080",
"2011_09_28_drive_0082",
"2011_09_28_drive_0086",
"2011_09_28_drive_0087",
"2011_09_28_drive_0089",
"2011_09_28_drive_0090",
"2011_09_28_drive_0094",
"2011_09_28_drive_0095",
"2011_09_28_drive_0096",
"2011_09_28_drive_0098",
"2011_09_28_drive_0100",
"2011_09_28_drive_0102",
"2011_09_28_drive_0103",
"2011_09_28_drive_0104",
"2011_09_28_drive_0106",
"2011_09_28_drive_0108",
"2011_09_28_drive_0110",
"2011_09_28_drive_0113",
"2011_09_28_drive_0117",
"2011_09_28_drive_0119",
"2011_09_28_drive_0121",
"2011_09_28_drive_0122",
"2011_09_28_drive_0125",
"2011_09_28_drive_0126",
"2011_09_28_drive_0128",
"2011_09_28_drive_0132",
"2011_09_28_drive_0134",
"2011_09_28_drive_0135",
"2011_09_28_drive_0136",
"2011_09_28_drive_0138",
"2011_09_28_drive_0141",
"2011_09_28_drive_0143",
"2011_09_28_drive_0145",
"2011_09_28_drive_0146",
"2011_09_28_drive_0149",
"2011_09_28_drive_0153",
"2011_09_28_drive_0154",
"2011_09_28_drive_0155",
"2011_09_28_drive_0156",
"2011_09_28_drive_0160",
"2011_09_28_drive_0161",
"2011_09_28_drive_0162",
"2011_09_28_drive_0165",
"2011_09_28_drive_0166",
"2011_09_28_drive_0167",
"2011_09_28_drive_0168",
"2011_09_28_drive_0171",
"2011_09_28_drive_0174",
"2011_09_28_drive_0177",
"2011_09_28_drive_0179",
"2011_09_28_drive_0183",
"2011_09_28_drive_0184",
"2011_09_28_drive_0185",
"2011_09_28_drive_0186",
"2011_09_28_drive_0187",
"2011_09_28_drive_0191",
"2011_09_28_drive_0192",
"2011_09_28_drive_0195",
"2011_09_28_drive_0198",
"2011_09_28_drive_0199",
"2011_09_28_drive_0201",
"2011_09_28_drive_0204",
"2011_09_28_drive_0205",
"2011_09_28_drive_0208",
"2011_09_28_drive_0209",
"2011_09_28_drive_0214",
"2011_09_28_drive_0216",
"2011_09_28_drive_0220",
"2011_09_28_drive_0222",
"2011_09_28_drive_0225",
"2011_09_29_drive_0004",
"2011_09_29_drive_0026",
"2011_09_29_drive_0071",
"2011_09_29_drive_0108",
"2011_09_30_drive_0016",
"2011_09_30_drive_0018",
"2011_09_30_drive_0020",
"2011_09_30_drive_0027",
"2011_09_30_drive_0028",
"2011_09_30_drive_0033",
"2011_09_30_drive_0034",
"2011_09_30_drive_0072",
"2011_10_03_drive_0027",
"2011_10_03_drive_0034",
"2011_10_03_drive_0042",
"2011_10_03_drive_0047",
"2011_10_03_drive_0058"]


def make_dataset():
    global number_list,TEMP_DIR
    number_list=[]
    for dataset in data_dirs:
        if dataset in test_dirs:
            continue
        data_year=dataset.split("_")[0]
        data_month=dataset.split("_")[1]
        data_date=dataset.split("_")[2]
        
        data_num="02"
        IMAGE_DIR=base_path+data_year+"_"+data_month+"_"+data_date+"/"+ dataset +"_sync/image_"+data_num+"/data/"
        
        file_names=[f.name for f in os.scandir(IMAGE_DIR) if not f.name.startswith('.')]
        
        OUTPUT_DIR1= TEMP_DIR+data_year+"_"+data_month+"_"+data_date+"/"+dataset+"_sync_"+data_num+'/image_02/data'
        
        if not os.path.exists(OUTPUT_DIR1+"/"):
            os.makedirs(OUTPUT_DIR1+"/")
            
        make_dataset1(OUTPUT_DIR1,file_names,dataset,IMAGE_DIR,data_num)
        
        OUTPUT_DIR2= TEMP_DIR+data_year+"_"+data_month+"_"+data_date+"/"+dataset+"_sync_"+data_num+'/image_03/data'
        
        if not os.path.exists(OUTPUT_DIR2+"/"):
            os.makedirs(OUTPUT_DIR2+"/")
            
        make_mask_images(OUTPUT_DIR2,file_names,dataset,IMAGE_DIR,data_num)
        
        
        data_num="03"
        IMAGE_DIR=base_path+data_year+"_"+data_month+"_"+data_date+"/"+ dataset +"_sync/image_"+data_num+"/data/"
        
        file_names=[f.name for f in os.scandir(IMAGE_DIR) if not f.name.startswith('.')]
        
        OUTPUT_DIR1= TEMP_DIR+data_year+"_"+data_month+"_"+data_date+"/"+dataset+"_sync_"+data_num+'/image_02/data'
        
        if not os.path.exists(OUTPUT_DIR1+"/"):
            os.makedirs(OUTPUT_DIR1+"/")
            
        make_dataset1(OUTPUT_DIR1,file_names,dataset,IMAGE_DIR,data_num)
        
        OUTPUT_DIR2= TEMP_DIR+data_year+"_"+data_month+"_"+data_date+"/"+dataset+"_sync_"+data_num+'/image_03/data'
        
        if not os.path.exists(OUTPUT_DIR2+"/"):
            os.makedirs(OUTPUT_DIR2+"/")
            
        make_mask_images(OUTPUT_DIR2,file_names,dataset,IMAGE_DIR,data_num)
        
        
        OUTPUT_TXT_FILE=TEMP_DIR+data_year+"_"+data_month+"_"+data_date+"/calib_cam_to_cam.txt"
        shutil.copyfile(INPUT_TXT_FILE, OUTPUT_TXT_FILE)
    
    
    run_all(file_names)
    TXT_RESULT_PATH=OUTPUT_DIR
    with open(TXT_RESULT_PATH+"/train.txt", mode='w') as f:
        f.write('\n'.join(number_list))
        
        
        
        
        

# In[35]:


def make_dataset1(OUTPUT_DIR1,file_names,dataset,IMAGE_DIR,data_num):
    for i in range(0,len(file_names)):        
        image_file=IMAGE_DIR + file_names[i]
        img = cv2.imread(image_file)
        img=cv2.resize(img,(WIDTH,HEIGHT))
        if not os.path.exists(OUTPUT_DIR1):
            os.makedirs(OUTPUT_DIR1)
        cv2.imwrite(OUTPUT_DIR1 + '/' + data_num + "_" + file_names[i] + '.jpg', img)


# In[36]:


def make_mask_images(OUTPUT_DIR2,file_names,dataset,IMAGE_DIR,data_num):
    for i in range(0,len(file_names)):   
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
        image=cv2.resize(image,(WIDTH,HEIGHT))
        
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        # Prepare black image
        mask_base = np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.uint8)
        after_mask_img = image.copy()
        color = (10, 10, 10) #white
        number_of_objects=len(r['masks'][0,0])
        mask_img=mask_base


        for j in range(0,number_of_objects):

            mask = r['masks'][:, :, j]

            mask_img = visualize.apply_mask(mask_base, mask, color,alpha=1)
        
            if not os.path.exists(OUTPUT_DIR2):
                os.makedirs(OUTPUT_DIR2)
        cv2.imwrite(OUTPUT_DIR2 + '/' + data_num + "_" + file_names[i] + '.jpg',mask_img)


# In[37]:


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    aseg = segimg[int(segimg.shape[0]*(half)):int(segimg.shape[0]*(1-half)), :]
    cy /= (1/middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128*a.shape[1]/a.shape[0]))
    x_scaling = float(wdt)/a.shape[1]
    y_scaling = 128.0/a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx*=x_scaling
    fy*=y_scaling
    cx*=x_scaling
    cy*=y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1]/416)
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]
    cseg = bseg[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c, cseg, fx, fy, cx, cy

def run_all(file_names):
    global number_list,OUTPUT_DIR,TEMP_DIR
    ct = 0
  
  
    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'

    for d in glob.glob(TEMP_DIR+'*/'):
        date = d.split('/')[-2]
        file_calibration = d + 'calib_cam_to_cam.txt'
        calib_raw = [get_line(file_calibration, 'P_rect_02'), get_line(file_calibration, 'P_rect_03')]

        for d2 in glob.glob(d + '*/'):
            DIR_NAME = d2.split('/')[-2]
            if not os.path.exists(OUTPUT_DIR + DIR_NAME +"/"):
                os.makedirs(OUTPUT_DIR + DIR_NAME +"/")
            print('Processing sequence', DIR_NAME)
            for subfolder in ['image_02/data']:
                ct = 1               
                calib_camera = calib_raw[0] if subfolder=='image_02/data' else calib_raw[1]
                folder = d2 + subfolder
                #files = glob.glob(folder + '/*.png')
                files = glob.glob(folder + '/*.jpg')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)
                for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                    imgnum = str(ct).zfill(10)
                    big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                    wct = 0

                    for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                        img = cv2.imread(files[j])
                        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                        zoom_x = WIDTH/ORIGINAL_WIDTH
                        zoom_y = HEIGHT/ORIGINAL_HEIGHT

                        # Adjust intrinsics.
                        calib_current = calib_camera.copy()
                        calib_current[0, 0] *= zoom_x
                        calib_current[0, 2] *= zoom_x
                        calib_current[1, 1] *= zoom_y
                        calib_current[1, 2] *= zoom_y

                        calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                        wct+=1
                    cv2.imwrite(OUTPUT_DIR  + DIR_NAME + '/' +  imgnum + '.png', big_img)
                    f = open(OUTPUT_DIR  + DIR_NAME + '/' + imgnum + '_cam.txt', 'w')
                    f.write(calib_representation)
                    f.close()
                    ct+=1
                    
            for subfolder in ['image_03/data']:
                ct = 1
                calib_camera = calib_raw[0] if subfolder=='image_02/data' else calib_raw[1]
                folder = d2 + subfolder
                #files = glob.glob(folder + '/*.png')
                files = glob.glob(folder + '/*.jpg')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)
                for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                    imgnum = str(ct).zfill(10)
                    big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                    wct = 0

                    for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                        img = cv2.imread(files[j])
                        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                        zoom_x = WIDTH/ORIGINAL_WIDTH
                        zoom_y = HEIGHT/ORIGINAL_HEIGHT

                        # Adjust intrinsics.
                        calib_current = calib_camera.copy()
                        calib_current[0, 0] *= zoom_x
                        calib_current[0, 2] *= zoom_x
                        calib_current[1, 1] *= zoom_y
                        calib_current[1, 2] *= zoom_y

                        calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                        wct+=1
                    #cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)
                    #f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
                    cv2.imwrite(OUTPUT_DIR   +DIR_NAME + '/' + imgnum + '-fseg.png', big_img)
                    f = open(OUTPUT_DIR  + DIR_NAME + '/' + imgnum + '_cam.txt', 'w')
                    number_list.append(DIR_NAME+" "+imgnum)
                    f.write(calib_representation)
                    f.close()
                    ct+=1


# In[38]:


make_dataset()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


