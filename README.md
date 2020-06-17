# PrepareDataForDFV
make dataset for depth from video

This code requires [MaskRCNN](https://github.com/matterport/Mask_RCNN) in the same dir.

## Input example (KITTI)

```script
nohup python GenDataKITTI.py &
```

 If you want to delete static frames designated in SfM Learner, please execute RemoveStaticFrames.ipynb after executing GenDataKITTI.py.

## Input example (Youtube videos)

First, please execute the initial part of ReadMovie.ipynb.

Then, please execute below.

```script
nohup python GenData.py &
```
