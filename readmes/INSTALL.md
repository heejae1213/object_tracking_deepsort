# Installation

Below is the environment the code was tested on (NVIDIA GPU and CUDA are required):
 
 * Ubuntu 18.04
 * Python 3.6
 * CUDA 10.1

Minimum requirements are listed in [requirements.txt](../requirements.txt)
You can use the command below to install all the necessary packages.
However, you may want to install each package by running individual commands with specified package versions  

```sh
pip install -r requirements.txt
```

Here are the commands I used to install required packages on Ubuntu 18.04 with CUDA 10.01 :

```sh
pip install opencv-python
pip install matplotlib
pip install scipy
pip install numpy
pip install --upgrade tensorflow
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install imgaug
pip install Pillow
pip install nonechucks
pip3 install scikit-learn==0.22.2
```



