# NCT project: Detector & Descriptor Outlier Rejection Evaluation for SLAM in MIS
The goal of this project is to find, impement and evaluate algorithms to reject false matches (outliers) from ORB feature matching.

## Required Python Libraries:
* NumPy
* matplotlib
* OpenCV
* imageio

### imageio .exr file support
The FreeImage backend needed for opening .exr-files (containing 3D coordinates of the sample data) is installed by running the following python code:
```
import imageio.plugins.freeimage
imageio.plugins.freeimage.download()
```

## Required environment for Flownet2.0
* Python3.7.9
* cuda 10.1
* pytorch 1.4.0
* numpy
* scipy
* scikit-image
* tensorboardX
* colorama, tqdm, setproctitle
### Installation  

    # get flownet2-pytorch source
    git clone https://github.com/NVIDIA/flownet2-pytorch.git
    cd flownet2-pytorch

    # install custom layers
    bash install.sh
 ### Converted Caffe Pre-trained Models
* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]
 ### Flownet Dataset
* http://sintel.is.tue.mpg.de/
## Required env for flowiz
* tqdm
* matplotlib
* numpy
* Eel
* Pillow
### Installation  

    # get flowiz source
    git clone https://github.com/georgegach/flowiz.git
    # usage
    cd flowiz
    python setup.py install --user
    python -m flowiz demo/flo/*.flo
    
## Sample Data Set:
The sample data in *./samples* was obtained from the following source: http://opencas.dkfz.de/video-sim2real/

## GitLab
Link: https://gitlab.com/docear/nct-project-feature-rejection

