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

You can add the dataset links into ```data/rgbd_tum_datasets.txt```. 

## Set Parameters

Create a yaml file to set the parameters.
Copy ```data/template.yaml``` if you want to make new one.

## Run on TUM-RGBD

```
cp data/template.yaml data/tumrgbd.yaml
```

Run ```run_kinectfusion_tumrgbd.cpp```.

It generates a pointcloud ply file and the pose text file.

You can evaluate the estimated pose by [evo](https://github.com/MichaelGrupp/evo).  

```
evo_rpe tum ../data/TUMRGBD/rgbd_dataset_freiburg1_xyz/groundtruth.txt rgbd_dataset_freiburg1_xyz_pose.txt -p --plot_mode=xyz
```

## Environment

* The environment we tested
  * Ubuntu 20.04
  * GTX 980ti 6GB
  * CUDA 11.2
  * OpenCV 4.5.2
