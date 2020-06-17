# PrepareDataForDFV
make dataset for depth from video

This code requires [MaskRCNN](https://github.com/matterport/Mask_RCNN) in the same dir.

## Input example (KITTI)

```script
nohup python GenDataKITTI.py &
```

## Input example (Youtube videos)

First, please execute beggining part of ReadMovie.ipynb.

Then, please execute below.

```script
nohup python GenData.py &
```
