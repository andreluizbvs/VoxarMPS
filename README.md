# Voxar MPS

  
  

> Fluid simulation framework based on the MPS method

  
  

![GUI & simulation output](https://lh3.googleusercontent.com/ZE8kEKhkJEvU-egRPZKXBxoNHCx9CC4ieHEqydhvALc8kjIEf6BgpgK-fmH6dwEn4a306E42hw4Rtg)

![Oil spilling](https://lh3.googleusercontent.com/0nM5gYtc49N9YH5Pv8XmEJt5doqyNWaqPsv3EDxKaIYaMLQ4aKadmHXI3tNCPGOZAisUT_2Tw0vv8Q "Oil spilling")

  

## Introduction

This is a fluid simulation framework based on the Moving Particle Semi-implicit method derived from results obtained in a M.Sc. thesis: "A fluid simulation system based on the MPS method". A set of papers (shown in the references section) was used as base to its development.

This framework also works as a tool, simulating pre-built scenarios that can be fine-tuned regarding its physical properties, compressibility approach, execution type (CPU/GPU) and others.

The creation of new simulation scenarios is not included and it is suggested as future work.

 - Sample [Input](https://drive.google.com/file/d/1_2_CBGs6ZkfPPBe5sQdbSbZ8js3dfHN3/view?usp=sharing) and [Output](https://drive.google.com/file/d/157k4OtAz3Ih-ZGoxwTlyX5RW140qf4Ur/view?usp=sharing) for one comprehensive test run of the 2D dam break problem.

## Features

The system makes possible to simulate fluids in different ways and approaches. A few possibilities are:

  

- Compressibility approach

	- Fully Incompressible

	- Weakly Compressible

- Numerical improvements in both compressibility approaches

- Type of execution

	- CPU - Sequential

	- CPU - OpenMP optimized

	- GPU - CUDA optimized

- Turbulence model

- Viscosity model

- Multiphase interaction (max. of 2 fluids)

	- Fine-tune density & viscosity of both fluids

- Time-step duration

  

## Requirements

  

- NVIDIA GPU

- Compute capability equal or greater than 6.1 (https://en.wikipedia.org/wiki/CUDA)

- CUDA 10.1 

- Visual Studio 2019 

## Installation

  

1. Update the NVIDIA GPU driver to the current version

2. Install CUDA 10.1 (https://developer.nvidia.com/cuda-downloads)

3. Install Visual Studio 2019 (Community, Professional or Enterprise) (https://visualstudio.microsoft.com/vs/)

4. Install Git and clone this repository

5. Build from source both Visual Studio projects, first the the VoxarMPS solution and then the GUI solution

6. Run GUI.exe to set the simulation parameters and start it

  

## Usage

  

![Graphical User interface](https://lh3.googleusercontent.com/k0kqTduLYezFVEYyErF-Y8sfkwEaz8n8bqSgs7SsN9wZSOKwHJnFZkd2ETjV6e8c1L2f-_5Wqo5uXw "Graphical User Interface")

  

- (1) Choose two or three dimensions simulations;

- (2) Type of execution: sequentially, parallelized through OpenMP or parallelized through CUDA;

- (3) Select from a set of previously built simulation scenarios;

- (4) Choose from two possible fluid compressibility approaches;

- By selecting Fully Incompressible, (5) and (6) will be enabled, which numerically improve calculations and stabilize fluid pressure;

- (7) Check to employ the SPS-LES turbulence model;

- (8) Can only be enabled if the chosen test scenario is a multiphase system. If checked, the second fluid in the simulation will present viscoplastic properties;

- In (9), (10), (11) and (12) the density and viscosity values of the two fluids in the simulation can be set;

- (13) Sets the time-step duration;

- (14) Sets how long the simulation will last in real-world time;

- (15) Generates the output VTU files with all kinds of particles present in the simulation;

- (16) Generates the output VTU files with only fluid particles information;

- (17) starts the generating the simulation;

- (18) allows the user to switch between PT-BR and EN-US languages.

  

To watch the simulation outcome any visualization software that reads VTU files can be used, such as [_ParaView_](https://www.paraview.org).


## Limitations

  

1. Number of particles in the simulation is limited by the size of available RAM

2. Only tested on Windows 10 using Visual Studio 2019 Enterprise and Community editions.

3. Only the following NVIDIA GPUs were used: GTX 1080 (Mobile) and GTX 1080 Ti

  

## File description

  

`main.cu`

- CUDA file that contains the main loop and calls for every utilized function of the algorithm, including CUDA kernels for the GPU runs

  

`functions.cu`

  

- CUDA file containing the implementations of the main routines and CUDA kernels used inside the main loop

  

`functions.cuh`

  

- All declarations of the routines and CUDA kernels implemented in `functions.cu`

  

`inOut.cpp`

  

- Implementations of functions that allow storing the output VTU files and also reading from these

  

`inOut.h`

  

- Declarations of the input and output routines from `inOut.cpp`

  

## Releases

  

- Version 1.0 released - 2019-XX-XX

  

## License

  

This program is distributed under GNU General Public License version 3 (GPLv3) license. Please see [LICENSE](https://github.com/andreluizbvs/VoxarMPS/blob/master/LICENSE) file.

  

## Authors

- André Luiz Buarque Vieira-e-Silva (albvs@cin.ufpe.br)
- Voxar Labs (voxarlabs@cin.ufpe.br) - https://www.cin.ufpe.br/~voxarlabs/

  

## References

Main references

- Vieira-e-Silva, André-Luiz et al. "A fluid simulation system based on the MPS method." _Computer Physics Communications_ (2019), Submitted.

- Shakibaeinia, Ahmad, and Yee-Chung Jin. "MPS mesh-free particle method for multiphase flows." _Computer Methods in Applied Mechanics and Engineering_ 229 (2012): 13-26.

- Gotoh, H. "Advanced particle methods for accurate and stable computation of fluid flows." _Frontiers of Discontinuous Numerical Methods and Practical Simulations in Engineering and Disaster Prevention_ (2013): 113.

- Koshizuka, Seiichi, and Yoshiaki Oka. "Moving-particle semi-implicit method for fragmentation of incompressible fluid." _Nuclear science and engineering_ 123.3 (1996): 421-434.
