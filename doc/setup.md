# Setup

This setup uses Python 3.7.7 in a conda environment on Ubuntu 18.04. The important packages are Tensorflow 2.1.0, [cyipopt](https://github.com/matthias-k/cyipopt), and cvxpy. [TEASER-plusplus](https://github.com/MIT-SPARK/TEASER-plusplus) and Gurobi 9.0.1 are optional. A Python requirements file of our setup (without Gurobi and TEASER-plusplus) is distributed, but shouldn't be required.

To setup everything except Gurobi and TEASER-plusplus using the requirements file:
```
$ conda env create -f environment.yml
```


The setup instructions (that worked around May 2020) without using the requirements file were:
```
$ conda create --name vio_calib
$ conda activate vio_calib
$ conda install tensorflow-gpu
$ conda install numpy scipy matplotlib
$ conda install -c conda-forge transforms3d
$ conda install -c conda-forge cyipopt
$ conda install -c conda-forge cvxpy
```


## Setup Gurobi
First, download and install Gurobi using the instructions in the [Gurobi Linux Quick Start](https://www.gurobi.com/documentation/9.0/quickstart_linux/index.html).

Then, the following commands will install Gurobi into the conda environment:
```
$ conda activate vio_calib
$ cd path/to/gurobi901/linux64/
$ python setup.py install
```


## Setup TEASER-plusplus
Clone and build TEASER into the conda environment:
```
$ conda activate vio_calib
$ git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
$ cd TEASER-plusplus
$ mkdir build
$ cd build
$ cmake ..
$ make teaserpp_python
$ cd python
$ pip install .
```
