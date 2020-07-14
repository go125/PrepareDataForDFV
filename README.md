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
nohup python GenData.py --base_path /home/ubuntu/data/all_video/ \
--ROOT_DIR ../Mask_RCNN \
--OUTPUT_DIR /home/ubuntu/data/tokushima_result20200312 \
--TEMP_DIR /home/ubuntu/data/train_data_example20200312/ &
```

## Input example (My City Report)

```script
nohup python GenData.py --base_path ../videos/ \
--ROOT_DIR ../Mask_RCNN \
--WIDTH 216 \
--HEIGHT 216 \
--OUTPUT_DIR ../out \
--TEMP_DIR ../tmpdir &
```
