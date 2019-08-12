/*

Voxar-MPS (Fluid simulation framework based on the MPS method)
Copyright (C) 2007-2011  Ahmad Shakibaeinia
Copyright (C) 2016-2019  Voxar Labs

Voxar-MPS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


*/

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#include <Windows.h>
#define STCK_SIZE 5000000000
#define NEIGHBORS 300
#define PI 3.1415926535897932
#define DIAG 2.0

//#define SHEAR_CAVITY // Turn on to test vorticity and fluid behavior through the Shear-driven Cavity problem
//#define WATER_DROP   // Turn on to test incompressibility through a free-falling water drop in a velocity field (debug)
//#define TestTURB     // Turn on to test turbulence through Rayleigh-Taylor instability
//#define OIL_SPILL     // Turn on to test multiphase oil spilling

//#define VISCOPLASTIC_F2 // Turn on to enable the Viscoplastic model

//#define TURBULENCE     // Turn on to enable SPS-LES Turbulence model

//#ifndef OIL_SPILL
//#define FI_MPS       // Turn on to enable Fully Incompressible version (if not defined the code runs the Weakly Compressible MPS)  
//#endif // OIL_SPILL

//#define MATRIX_OPT     // Turn on to enable numerical precision optimization for the FI_MPS (New laplacian model) -> Tende a vazar
//#define SOURCE_OPT     // Turn on to enable numerical precision optimization for the FI_MPS (New derivative model & ECS)

//#define _WITHOMP       // Turn on to enable CPU optimization - OpenMP (better only to turn on with GPU turned off)
//#define GPU          // Turn on to enable GPU optimization - CUDA (turn off _WITHOMP if this is turned on)
//#define CUSP         // Turn on to enable the usage of the CUSP library to solve the PPE in GPU ------> BUGGED <------

//#define _3D          // Turn on to enable 3D simulation -> z-coordinate

//#define PRINTALL	    // Turn on to output all particles
//#define PRINTFLUID      // Turn on to output only fluid particles
//#define PRINTBOUNDARY // Turn on to output only boundary particles

using namespace std;

void saveParticles(int TP, int Tstep, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double* pnew, int* PTYPE, int dim);

void saveFluidParticles(int TP, int FP, int Tstep, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double* pnew, int* PTYPE, int dim);

void getVTU(double*& x, double*& y, double*& z, double*& unew, double*& vnew, double*& wnew, double*& pnew, int*& PTYPE, int& TP, int& FP, int& GP, int& WP,
	double& Xmin, double& Xmax, double& Ymin, double& Ymax, double& Zmin, double& Zmax);

void getTP(int& TP);
