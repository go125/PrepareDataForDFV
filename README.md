# PrepareDataForDFV
make dataset for depth from video

This code requires [MaskRCNN](https://github.com/matterport/Mask_RCNN) in the same dir.

## Input example (KITTI)

```script
nohup python GenDataKITTI.py &
```

 If you want to delete static frames designated in SfM Learner, please execute RemoveStaticFrames.ipynb after executing GenDataKITTI.py.
 
 
## Input example (KITTI_gray)

```script
nohup python GenDataKITTI_gray.py &
```

## Input example (Youtube videos)

First, please execute the initial part of ReadMovie.ipynb.

Then, please execute below.

```script
nohup python GenData.py --base_path /home/ubuntu/data/all_video/ \
--ROOT_DIR ../Mask_RCNN \
--OUTPUT_DIR /home/ubuntu/data/tokushima_result20200312 \
--TEMP_DIR /home/ubuntu/data/train_data_example20200312/ &
```



## Input example (Stereo Camera)

```script
nohup python GenData.py --base_path ../all_video/ \
--ROOT_DIR ../Mask_RCNN \
--WIDTH 416 \
--HEIGHT 128 \
--OUTPUT_DIR ../out \
--TEMP_DIR ../tmpdir &
```


### Data structure example

```
..
├── Mask_RCNN
├── PrepareDataForDFV
│   ├── GenData.py
│   ├── calib_cam_to_cam.txt
│   └── etc
└── videos
    ├── video1
    │   ├── frame1.jpg
    │   ├── frame2.jpg
    │   └── frame3.jpg
    ├── video2
    │   ├── frame1.jpg
    │   ├── frame2.jpg
    │   └── frame3.jpg
    └── etc
```
