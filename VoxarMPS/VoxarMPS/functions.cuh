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

#include <tchar.h>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <tuple>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cusparse.h"
//#include "cusparse_v2.h"
#include "inOut.h"
#include "cublas_v2.h"
#include "cusolverSp.h"
#include "neighbour_parallel.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"


#include <stdlib.h>
#include <assert.h>

#include <omp.h>

using namespace std;

__host__ __device__ double W(double R, int KTYPE, int dim, double re);

__host__ __device__ double diff(int i, int j, double* x);

__host__ __device__ double dist2d(int i, int j, double* x, double* y);

__host__ __device__ double dist3d(int i, int j, double* x, double* y, double* z);

__host__ __device__ double dist2d(int i, int j, double* xstar, double* ystar);

__host__ __device__ double dist3d(int i, int j, double* xstar, double* ystar, double* zstar);

double* PNUM(int I, int ktype, int dim, int TP, double re, int** neighb, double* x, double* y, double* z, double Ncorrection, double* n, double* MAX, double* pnew, double* p, bool loopstarted, string codeOpt);

double* PNUMSTAR(int ktype, int dim, int TP, double re, int** neighb, double* xstar, double* ystar, double* zstar, double* nstar, string codeOpt);

void CGM(int TP, double* b, int IterMax, double MAXresi, double** poiss, int** neigh, int* bcon, double* x, int imax, double** ic, double dt, double eps, int imin, string codeOpt);

void MATRIX(double re, int Method, int FP, int WP, int TP, double* x, double* y, double* z, double coll, int KTYPE, int* PTYPE, double correction,
	double* nstar, double BETA, double n0, double* pnew, double Rho, double relaxp, double lambda, double DT,
	double* p, double* n, int GP, int dim, int** neigh, double** poiss, int* bcon, double* source, double* unew, double* vnew, double* wnew, bool matopt, bool srcopt, string codeOpt);

void BC(int slip, int TP, int GP, int WP, int* PTYPE, int I, int** neighb, double* x, double* y, double DL, double* v, double* vstar, double* vnew,
	double* u, double* ustar, double* unew, double* p, double* pnew, string codeOpt);

double* PHATCALC(int TP, int** neighb, double* pnew, double* phat, string codeOpt);

void PRESSGRAD(int GP, int WP, int KHcorrection, int TP, double* pnew, int** neighb, double* xstar, double* ystar, double* zstar, double* phat, int KTYPE, double re, double* RHO, double* ustar, double* vstar, double* wstar, double DT,
	double* unew, double* vnew, double* wnew, double relaxp, double n0, double VMAX, int dim, string codeOpt, double *nstar);

int** NEIGHBOR(double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, double re, double DELTA, int TP, double* x, double* y, double* z, int** neighb, int dim, string codeOpt);

void COLLISION2(int TP, double MINdistance, int* PTYPE, double Rho1, double Rho2, int** neighb, double CC, double* unew, double* vnew, double* wnew, double* x, double* y, double* z, double DT, int dim, string codeOpt);

double DTcalculation(double c0, double c01, double c02, double* DT, double DT_MAX, double COURANT, double DL);

void SPS(double re, int TP, int GP, int** neighb, double* x, double* y, double* z, double coll, int KTYPE, double* unew, double* vnew, double* wnew,
	double n0, double* NEUt, double Cs, double DL, double* TURB1, double* TURB2, double* TURB3, int dim, string codeOpt);

void BOUNDARIE(double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL, double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, int I, double CC, int dim, string test);

void EULERINTEGRATION(int GP, int WP, int TP, double DT, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL, double Xmax, double Xmin, double Ymax, double Ymin,
	double Zmax, double Zmin, int I, double CC, int dim, string test, string codeOpt);

double* VISCOSITY(double re, int TP, int Fluid2_type, int* PTYPE, double* MEU, double NEU1, double NEU2, double Rho1, double Rho2, int** neighb, double* x, double* y, double* z,
	int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* C, double PHI, int I, double cohes, double II, double yield_stress, double* phat, double MEU0, double N, int dim, string codeOpt);

void PREDICTION(double re, double* xstar, double* ystar, double* zstar, double* ustar, double* vstar, double* wstar, double* u, double* v, double* w, int TP, int* PTYPE, double* MEU,
	double Rho1, double Rho2, int** neighb, double* x, double* y, double* z, int KTYPE, double n0, double* phat, double* pnew, double gx, double gy, double gz, double DT, double* NEUt, double lambda,
	double* TURB1, double* TURB2, double* TURB3, double relaxp, double* RHO, double* SFX, double* SFY, int dim, string codeOpt);

double** INCDECOM(int TP, int* bcon, int** neigh, double** poiss, double** ic, string codeOpt);

int* BCON(int* PTYPE, int* bcon, double* n, double n0, double dirichlet, int TP, string codeOpt);

int* CHECKBCON(int* check, int* bcon, int** neigh, int TP, string codeOpt);

void PRESSURECALC(int Method, int GP, int FP, int WP, int TP, int* PTYPE, double c0, double c01, double c02, double Rho1, double Rho2, double* C, double* nstar, double BETA, double n0, double* pnew, double PMIN, double PMAX,
	int IterMax, double MAXresi, double re, double* x, double* y, double* z, double coll, int KTYPE, double correction, double Rho, double relaxp, double lambda,
	double DT, double* p, double* n, int dim, int** neigh, double** poiss, int* bcon, double* source, double** ic, int imin, int imax, double eps, double* unew, double* vnew, double* wnew, bool matopt, bool srcopt, string codeOpt);

double* V_FRACTION(double re, int Fraction_method, int TP, int** neighb, int* PTYPE, double* C, int KTYPE, double* x, double* y, double* z, int dim, string codeOpt);


void PREPDATA(int TP, int FP, double* x, double* y, double* z, double* u, double* v, double* w, double* p, double* unew, double* vnew, double* wnew, double* pnew, double Xmin, double Ymin, double Xmax, double Ymax, double Zmin, double Zmax, int dim, string test, string codeOpt);

void neighbor2gpu(int TP, double Xmin, double Ymin, double Zmin, double Xmax, double Ymax, double Zmax, double re, double DELTA, int ncx, int ncy, int tnc, double* x, double* y, double* z,
	int* Iend, int* Ista, int* ip, int dim, string codeOpt);

//-------------------------------CUDA-----------------------------------//

__global__ void pnum(int offset, int ktype, int dim, int TP, double re, int* neighb, double* x, double* y, double* z, double Ncorrection, double* n, double* MAX, double* pnew, double* p,
	bool loopstarted);

__global__ void pnumstar(int offset, int KTYPE, int DIM, int TP, double re, int* d_neighb, double* d_xstar, double* d_ystar, double* d_zstar, double* d_nstar);

__global__ void neighbor1(int offset, double Xmax, double Xmin, double Ymax, double Ymin, double re, double DELTA, int TP, double* d_x, double* d_y, int* d_neighb, int* d_Ista,
	int* d_Iend, int* d_nc, int* d_ip);

// __global__ void neighbor2(int offset, double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, double re, double DELTA, int TP, int TP, double* d_x, double* d_y, double* d_z,
//	int *d_neighb, int *d_Ista, int *d_Iend, int *d_nc, int *d_ip, int dim);

__global__ void neighbor3(int offset, double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, double re, double DELTA, int TP, double* d_x, double* d_y, double* d_z,
	int* d_neighb, int* d_Ista, int* d_Iend, int* d_nc, int* d_ip, int dim);

__global__ void eulerIntegration(int offset, int GP, int WP, int TP, double DT, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL,
	double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, int i, double CC, int dim);

__global__ void dtCalculation(int offset, double c0, double c01, double c02, double* DT, double DT_MAX, double COURANT, double DL);

__global__ void pressGrad(int offset, int GP, int WP, int KHcorrection, int TP, double* pnew, int* neighb, double* xstar, double* ystar, double* zstar, double* phat, int KTYPE, double re,
	double* RHO, double* ustar, double* vstar, double* wstar, double DT, double* unew, double* vnew, double* wnew, double relaxp, double n0, double VMAX, int dim);

__global__ void phatCalc(int offset, int TP, int* neighb, double* pnew, double* phat);

__global__ void collision2(int offset, int TP, double MINdistance, int* PTYPE, double Rho1, double Rho2, int* neighb, double CC, double* unew, double* vnew, double* wnew, double* x, double* y, double* z,
	double DT, int dim);

__global__ void bc(int offset, int slip, int TP, int GP, int WP, int* PTYPE, int I, int* neighb, double* x, double* y, double DL, double* v, double* vstar, double* vnew, double* u,
	double* ustar, double* unew, double* p, double* pnew);

__global__ void prepData(int offset, int TP, double* x, double* y, double* z, double* u, double* v, double* w, double* p, double* unew, double* vnew, double* wnew, double* pnew, double Xmin, double Ymin,
	double Zmin, double Xmax, double Ymax, double Zmax, int dim);

__global__ void pressureCalcWC(int offset, int* PTYPE, double c0, double c01, double c02, double Rho1, double Rho2, double* C, double* nstar, double BETA, double n0, double* pnew, double PMIN, double PMAX,
	double Rho);

__global__ void prediction(int offset, double re, double* xstar, double* ystar, double* zstar, double* ustar, double* vstar, double* wstar, double* u, double* v, double* w, int TP, int* PTYPE,
	double* MEU, double Rho1, double Rho2, int* neighb, double* x, double* y, double* z, int KTYPE, double n0, double* phat, double* pnew, double gx, double gy, double gz, double DT,
	double* NEUt, double lambda, double* TURB1, double* TURB2, double* d_TURB3, double relaxp, double* RHO, double* SFX, double* SFY, int dim);

__global__ void viscosity(int offset, double re, int TP, int Fluid2_type, int* PTYPE, double* MEU, double NEU1, double NEU2, double Rho1, double Rho2, int* neighb, double* x, double* y, double* z,
	int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* C, double PHI, int I, double cohes, double II, double yield_stress, double* phat, double MEU0, double N,
	double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim);

__global__ void volFraction(int offset, double re, int Fraction_method, int TP, int* neighb, int* PTYPE, double* C, int KTYPE, double* x, double* y, double* z, int dim);

__global__ void turb1(int offset, double re, int* neighb, double* x, double* y, double* z, double coll, int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* NEUt, double Cs,
	double DL, double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim);

__global__ void turb2(int offset, double re, int* neighb, double* x, double* y, double* z, double coll, int KTYPE, double n0, double* NEUt, double* TURB1, double* TURB2, double* TURB3,
	double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim);

__global__ void bconCalc(int offset, int* PTYPE, int* bcon, double* n, double n0, double dirichlet, int TP);

__global__ void sourceCalc(int offset, int TP, int* PTYPE, int* bcon, double* nstar, double n0, double DT, double* source, int* neigh, double* x, double* y, double* z,
	double* unew, double* vnew, double* wnew, double re, int dim, bool srcopt);

__global__ void matrixCalc(int offset, double re, double* x, double* y, double* z, int KTYPE, double* nstar, double n0, double Rho, double lambda, double DT, int dim, int* neigh, double* poiss, int* bcon, bool matopt);

__global__ void incdecom(int offset, int TP, int* bcon, int* neigh, double* poiss, double* ic);

__global__ void cgm1(int offset, int TP, int* bcon, double* s, int* neigh, double* poiss, double* x);

__global__ void cgm2(int offset, double* r, double* b, double* s);

__global__ void cgmFS(int offset, double* ic, double* q, double* aux, int* bcon, int* neigh, int TP);

__global__ void cgmBS(int offset, double* ic, double* q, double* aux, int* bcon, int* neigh, int TP);

__global__ void cgm5(int offset, double* r, double* q, int* bcon, float* rqo);

__global__ void cgm6(int offset, int TP, int* bcon, double* s, int* neigh, double* poiss, double* p);

__global__ void cgm7(int offset, double* p, double* s, int* bcon, float* ps);

__global__ void cgm8(int offset, double* x, double* p, double* aa);

__global__ void cgm9(int offset, double* r, double* s, double* aa);

__global__ void cgm10(int offset, double* r, double* q, int* bcon, float* rqn);

__global__ void cgm11(int offset, double* p, double* q, double* bb);

__global__ void cgm12(int offset, double* r, int* bcon, double* err1, int* j, double eps, double dt);