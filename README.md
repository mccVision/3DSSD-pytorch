# 3DSSD-pytorch
This is the unofficial implementation of [3DSSD](https://arxiv.org/abs/2002.10187) in pytorch. The official implementation is [here](https://github.com/Jia-Research-Lab/3DSSD.git).  
This implementation has some bugs, I just move the repo here and try to fix the bugs. It can't reach the mAP of 3DSSD. 
## Requirements
+ Ubuntu 18.04
+ Python 3.6
+ pytorch 1.4
+ CUDA 10.1 & CUDNN 7.5

## Preparation
1. Clone this repository
    ```angular2html
    git clone https://github.com/mccVision/3DSSD-pytorch.git
    ```
2. Install the Python dependencies  
    ```
    pip install -r requirements.txt
    ```
3. Compile and install library.   
   The functions mostly are from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git), F-FPS code comes from 3DSSD.
    ```
    sh compile_op.sh
    ```
4. Data Preparation  
I organize KITTI dataset just like [OpenPCDet GETTING_STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md), please download the official [KITTI detaset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
   
## Train
   ```angular2html
   cd tools
   python train.py --cfg ../configs/kitti/car_cfg.py
```
## Performance
I just get the results as below. It exists some gaps compared to the official results. And I can't figure out what is wrong with my code. 
   ```angular2html
   Car AP_R40@0.70, 0.70, 0.70:
   bbox AP:96.5015, 93.1626, 88.1580
   bev  AP:93.0096, 89.1966, 84.2649
   3d   AP:90.7322, 80.3654, 75.4788
   aos  AP:96.47, 93.09, 88.04
   Car AP_R40@0.70, 0.50, 0.50:
   bbox AP:96.5015, 93.1626, 88.1580
   bev  AP:96.4874, 95.4035, 90.4071
   3d   AP:96.4691, 95.3144, 90.3186
   aos  AP:96.47, 93.09, 88.04
```
## Acknowledgment
This repo borrows code from several repos, like [VoteNet](https://github.com/facebookresearch/votenet.git), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git), [3DSSD](https://github.com/Jia-Research-Lab/3DSSD.git), [Det3D](https://github.com/poodarchu/Det3D.git) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d.git).
