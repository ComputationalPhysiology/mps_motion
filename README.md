![CI](https://github.com/ComputationalPhysiology/mps_motion/workflows/CI/badge.svg)

# MPS Motion Tracking

Library for tracking motion in cardiac mps data


* Source code: https://github.com/ComputationalPhysiology/mps_motion

## Installation

You can install the software with pip. Clone the repo, change director to the root of the repo and do
```
python3 -m pip install install .
```
There is also a `Makefile` in this repo, and the same instructions can be executed using the command `make install`.

Alternatively you can install it directly from Github
```
python3 -m pip install git+https://github.com/ComputationalPhysiology/mps_motion
```

### Development installations
Developers should install some extra dependencies
```
python3 -m pip install install -e ".[dev,docs,test]"
```
as well as the pre-commit hook
```
pre-commit install
```
Alternatively, you can use the `Makefile` and hit `make dev`.

## Getting started


### Command line interface
You can analyse a dataset used the command line interface as follows
```
python -m mps_motion PointH4A_ChannelBF_VC_Seq0018.nd2
```
Here `PointH4A_ChannelBF_VC_Seq0018.nd2` is an example file containing cardiac cell data.
Note that in order to read these data you need to also install the `mps` package.
To see all available options for the cli you can do
```
python -m mps_motion --help
```

### Computing displacement and strain
```python
import matplotlib.pyplot as plt # For plotting
import mps # Package to load data
import mps_motion as mmt # Package for motion analysis

# Load raw data from Nikon images
data = mps.MPS("data.nd2")
print(data.info) # {'num_frames': 267, 'dt': 11.92, 'time_unit': 'ms',
# 'um_per_pixel': 0.325, 'size_x': 2044, 'size_y': 1174}
opt_flow = mmt.OpticalFlow(data, flow_algorithm="farneback", reference_frame=0)
u = opt_flow.get_displacements(scale=0.4)
print(u) # VectorFrameSequence((817, 469, 267, 2), dx=0.8125, scale=0.4)
u_norm = u.norm()
print(u_norm) # FrameSequence((817, 469, 267), dx=0.8125, scale=0.4)
# Compute mean of the norm of all pixels
u_norm_mean = u_norm.mean().compute()
plt.figure()
plt.plot(data.time_stamps, u_mean_norm)

mech = mmt.Mechancis(u=u, t=data.time_stamps)
# Green-Lagrange strain
print(mech.E) # TensorFrameSequence((817, 469, 267, 2, 2), dx=0.8125, scale=0.4)

# Plot the X-component of the Green-Lagrange strain
plt.figure()
plt.plot(data.time_stamps, mech.E.x.mean().compute())
plt.show()
```



## Useful links:

- <https://nanonets.com/blog/optical-flow/>
- https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
- <https://vision.middlebury.edu/flow/data/>
- <https://github.com/chuanenlin/optical-flow>
- <https://github.com/tsenst/CrowdFlow>
- <https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/>



## Authors

- Henrik Finsberg henriknf@simula.no
