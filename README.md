# Gaussian Haircut: Human Hair Reconstruction with Strand-Aligned 3D Gaussians

[**Paper**](https://arxiv.org/abs/2409.14778) | [**Project Page**](https://eth-ait.github.io/GaussianHaircut/)

This repository contains an official implementation of Gaussian Haircut, a strand-based hair reconstruction approach for monocular videos.

## Getting started

1. **Install CUDA 11.8**

   Follow intructions on https://developer.nvidia.com/cuda-11-8-0-download-archive

   Make sure that
     -   PATH includes <CUDA_DIR>/bin
     -   LD_LIBRARY_PATH includes <CUDA_DIR>/lib64

   The environment was tested only with this CUDA version

2. **Install Blender 3.6** in order to create strand visualizations

   Follow instructions on https://www.blender.org/download/lts/3-6

3. **Close the repo and run the install script**

    ```bash
    git clone git@github.com:eth-ait/GaussianHaircut.git
    cd GaussianHaircut
    chmod +x ./install.sh
    ./install.sh
    ```

## Reconstruction

1. **Record a monocular video**

   Tips: use examples on the project as references; introduce as little motion blur as possible; be mindful that off-the-shelf segmentation and pose estimations networks will be run for the captured frames

2. **Setup a directory for the reconstructed scene**

   Put the video file in it and rename to raw.mp4

3. **Run the script**

   ```bash
   export PROJECT_DIR="[/path/to/]GaussianHaircut"
   export BLENDER_DIR="[/path/to/blender/folder/]blender"
   DATA_PATH="[path/to/scene/folder]" ./run.sh
   ```

The script performs data pre-processing, reconstruction and generation of visualizations similar to the ones reported in the paper.

## License

This code is based on 3D Gaussian Splatting project, please refer to LICENSE_3DGS for terms and conditions. The rest of the code is distributed under CC BY-NC-SA 4.0.

If you find this code useful in your project, make sure to cite the papers below.

## Citation

```
@inproceedings{zakharov2024gh,
   title = {Human Hair Reconstruction with Strand-Aligned 3D Gaussians},
   author = {Zakharov, Egor and Sklyarova, Vanessa and Black, Michael J and Nam, Giljoo and Thies, Justus and Hilliges, Otmar},
   booktitle = {European Conference of Computer vision (ECCV)},
   year = {2024}
} 
```

## Links

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

- [Neural Haircut](https://github.com/SamsungLabs/NeuralHaircut): FLAME fitting pipeline, strand prior and hairstyle diffusion prior

- [HAAR](https://github.com/Vanessik/HAAR): hair upsampling

- [Matte-Anything](https://github.com/hustvl/Matte-Anything): hair and body segmentation

- [PIXIE](https://github.com/yfeng95/PIXIE): initialization for FLAME fitting

- [Face-Alignment](https://github.com/1adrianb/face-alignment), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): keypoints detection for FLAME fitting
