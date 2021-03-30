# Learned Uncertainty Calibration

A collection of Python scripts that connect to SLAM systems and analyze their
covariance estimates on various datasets.

Currently supports [XIVO](https://github.com/ucla-vision/xivo) with the
following datasets:
- [TUMVI VIO](https://vision.in.tum.de/data/datasets/visual-inertial-dataset)
- XIVO dataset distributed with XIVO


## Setup

1. Install dependencies: numpy, scipy, matplotlib.
2. Build XIVO. Ensure that cmake uses the same version of Python 3.
3. Set the environment variable `XIVO_ROOT` to the root of XIVO source code.
4. Follow instructions in [this file](doc/setup.md)


## Replicating the paper

Commands for executing the experiments in ``Learned Uncertainty Calibration for Visual Inertial Localization" are [here](doc/icra2021.md).
