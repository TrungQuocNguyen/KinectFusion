# KinectFusion

This repository is a re-implementation of KinectFusion paper, which is a part of group project for 3D Scanning and Motion Capture at TUM
KinectFusion paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
![plot](figs/output_kf.png)

## Dependencies

* Eigen
* CUDA
* OpenCV
  * WITH_CUDA: ON
  * WITH_VTK: ON

## Download Datasets

```
sh data/download_tumrgbd.sh
```

You can add dataset links into ```data/rgbd_tum_datasets.txt```. 

## Set Parameters

Make yaml file to set parameters.
Copy ```data/template.yaml``` if you want to make new one.

## Run on TUM-RGBD

run ```run_kinectfusion_tumrgbd.cpp```

You can change the parameters of ```data/tumrgbd.yaml``.

It generates pointcloud ply file and pose text file.
You can evaluate poses. 

