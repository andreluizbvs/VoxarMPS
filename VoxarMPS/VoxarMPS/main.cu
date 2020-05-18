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

#include "functions.cuh"
#include "inOut.h"


using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static const char* _cusparseGetErrorEnum(cusparseStatus_t error)
{
	switch (error)
	{

	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char* file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs", __FILE__, __LINE__, err, \
			_cusparseGetErrorEnum(err)); \
			cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

//profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR   100
#define DBICGSTAB_EPS           1.E-14f //9e-2

#define CLEANUP()                       \
do {                                    \
    if (x)          free (x);           \
    if (f)          free (f);           \
    if (r)          free (r);           \
    if (rw)         free (rw);          \
    if (p)          free (p);           \
    if (pw)         free (pw);          \
    if (s)          free (s);           \
    if (t)          free (t);           \
    if (v)          free (v);           \
    if (tx)         free (tx);          \
    if (Aval)       free(Aval);         \
    if (AcolsIndex) free(AcolsIndex);   \
    if (ArowsIndex) free(ArowsIndex);   \
    if (Mval)       free(Mval);         \
    if (devPtrX)    checkCudaErrors(cudaFree (devPtrX));                    \
    if (devPtrF)    checkCudaErrors(cudaFree (devPtrF));                    \
    if (devPtrR)    checkCudaErrors(cudaFree (devPtrR));                    \
    if (devPtrRW)   checkCudaErrors(cudaFree (devPtrRW));                   \
    if (devPtrP)    checkCudaErrors(cudaFree (devPtrP));                    \
    if (devPtrS)    checkCudaErrors(cudaFree (devPtrS));                    \
    if (devPtrT)    checkCudaErrors(cudaFree (devPtrT));                    \
    if (devPtrV)    checkCudaErrors(cudaFree (devPtrV));                    \
    if (devPtrAval) checkCudaErrors(cudaFree (devPtrAval));                 \
    if (devPtrAcolsIndex) checkCudaErrors(cudaFree (devPtrAcolsIndex));     \
    if (devPtrArowsIndex) checkCudaErrors(cudaFree (devPtrArowsIndex));     \
    if (devPtrMval)       checkCudaErrors(cudaFree (devPtrMval));           \
    if (stream)           checkCudaErrors(cudaStreamDestroy(stream));       \
    if (cublasHandle)     checkCudaErrors(cublasDestroy(cublasHandle));     \
    if (cusparseHandle)   checkCudaErrors(cusparseDestroy(cusparseHandle)); \
    fflush (stdout);                                    \
} while (0)


static void gpu_pbicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m, int n, int nnz,
	const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */
	double* a, int* ia, int* ja,
	const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
	double* vm, int* im, int* jm,
	cusparseSolveAnalysisInfo_t info_l, cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
	double* f, double* r, double* rw, double* p, double* pw, double* s, double* t, double* v, double* x,
	int maxit, double tol, double ttt_sv)
{
	double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
	double nrmr, nrmr0;
	rho = 0.0;
	double zero = 0.0;
	double one = 1.0;
	double mone = -1.0;
	int i = 0;
	int j = 0;
	double ttl, ttl2, ttu, ttu2, ttm, ttm2;
	double ttt_mv = 0.0;

	//WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable ttt_sv)

	//compute initial residual r0=b-Ax0 (using initial guess in x)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	checkCudaErrors(cudaDeviceSynchronize());
	ttm = second();
#endif

	checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x, &zero, r));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	cudaDeviceSynchronize();
	ttm2 = second();
	ttt_mv += (ttm2 - ttm);
	//printf("matvec %f (s)\n",ttm2-ttm);
#endif
	checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
	checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
	//copy residual r into r^{\hat} and p
	checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
	checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
	//printf("gpu, init residual:norm %20.16f\n",nrmr0); 

	for (i = 0; i < maxit; ) {
		rhop = rho;
		checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));

		if (i > 0) {
			beta = (rho / rhop) * (alpha / omega);
			negomega = -omega;
			checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, v, 1, p, 1));
			checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p, 1));
			checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r, 1, p, 1));
		}
		//preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_l, p, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl2 = second();
		ttu = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_u, t, pw));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttu2 = second();
		ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
		//printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif

		//matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm = second();
#endif

		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, pw, &zero, v));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm2 = second();
		ttt_mv += (ttm2 - ttm);
		//printf("matvec %f (s)\n",ttm2-ttm);
#endif

		checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, v, 1, &temp));
		alpha = rho / temp;
		negalpha = -(alpha);
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &negalpha, v, 1, r, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha, pw, 1, x, 1));
		checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

		if (nrmr < tol * nrmr0) {
			j = 5;
			break;
		}

		//preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_l, r, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl2 = second();
		ttu = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_u, t, s));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttu2 = second();
		ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
		//printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif
		//matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm = second();
#endif

		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, s, &zero, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm2 = second();
		ttt_mv += (ttm2 - ttm);
		//printf("matvec %f (s)\n",ttm2-ttm);
#endif

		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, r, 1, &temp));
		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &temp2));
		omega = temp / temp2;
		negomega = -(omega);
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega, s, 1, x, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, t, 1, r, 1));

		checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

		if (nrmr < tol * nrmr0) {
			i++;
			j = 0;
			break;
		}
		i++;
	}

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	printf("gpu total solve time %f (s), matvec time %f (s)\n", ttt_sv, ttt_mv);
#endif
}

__global__ void setDensity(int offset, double* d_RHO, const int TP, int* d_PTYPE, double Rho1, double Rho2)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (d_PTYPE[I] == 2) d_RHO[I] = Rho2;
	else d_RHO[I] = Rho1;
}

__global__ void aaKernel(double* d_aa, float* d_rqo, float* d_ps)
{
	d_aa[0] = d_rqo[0]/d_ps[0];
}

__global__ void bbKernel(double* d_bb, float* d_rqn, float* d_rqo)
{
	d_bb[0] = d_rqn[0] / d_rqo[0];
}

__global__ void pnewSet(int offset, double* pnew, double PMIN, double PMAX)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (pnew[I] < PMIN)
		pnew[I] = PMIN;

	if (pnew[I] > PMAX || pnew[I] * 0.0 != 0.0)						//		   Pressure assuming non-numeric values
		pnew[I] = PMAX;
}



int main(int argc,													//		   Number of strings in array argv  
	char* argv[],													//		   Array of command-line argument strings  
	char* envp[])													//		   Array of environment variable strings  
{
	int argNumber = 16;
	for (int i = 0; i < argc; i++)cout << string(argv[i]) << endl;
	if (argc - 1 != argNumber)
	{
		std::cout << "Unexpected number of arguments: " << argc - 1 << " (expected " << argNumber << ")" << endl;
		std::cout << endl << "Usage: MPS_shak_GPU.exe [test] [viscoplastic (in second fluid)] [turbulence] [compressibility] [matrix optimization] " <<
			"[source optimization] [performance optimization] [dimensions] [print all particles] [print only fluid] [densityfluid1] [viscosityfluid1] [densityfluid2] [viscosityfluid2] [totalsimulationtime] [timestepduration]" << endl;
		std::cout << endl << "Example: MPS_shak_GPU.exe oil novisc turb comp matopt srcopt gpu 2d no printfluid 1000.0 0.000001 500.0 0.001 1 0.005" << endl;
		system("pause");
		exit(1);
	}

	string test;
	bool visc;
	bool turb;
	bool compressible;
	bool matopt;
	bool srcopt;
	string codeOpt;
	bool threedim;
	bool outAll;
	bool outFluid;
	double stepsToCalcNeigh = 1.0;									//		   How many steps until the next neighborhood update

	if (string(argv[1]) == "oil")
	{
		test = string(argv[1]);
	}
	else if (string(argv[1]) == "drop")
	{
		test = string(argv[1]);
	}
	else if (string(argv[1]) == "shear")
	{
		test = string(argv[1]);
	}
	else if (string(argv[1]) == "turb")
	{
		test = string(argv[1]);
	}
	else
	{
		test = "dam";
	}

	if (string(argv[2]) == "visc" || string(argv[2]) == "viscosity" || string(argv[2]) == "viscoplasticity" || string(argv[2]) == "true" || string(argv[2]) == "yes")
	{
		visc = true;
	}
	else
	{
		visc = false;
	}

	if (string(argv[3]) == "turb" || string(argv[3]) == "turbulence" || string(argv[3]) == "true" || string(argv[3]) == "yes")
	{
		turb = true;
	}
	else
	{
		turb = false;
	}

	if (string(argv[4]) == "comp" || string(argv[4]) == "compressible" || string(argv[4]) == "true" || string(argv[4]) == "yes")
	{
		compressible = true;
	}
	else
	{
		compressible = false;

		if (test == "oil") compressible = true;
	}

	if (string(argv[5]) == "matopt" || string(argv[5]) == "matrixoptimization" || string(argv[5]) == "true" || string(argv[5]) == "yes")
	{
		matopt = true;
	}
	else
	{
		matopt = false;
	}

	if (string(argv[6]) == "srcopt" || string(argv[6]) == "sourceoptimization" || string(argv[6]) == "true" || string(argv[6]) == "yes")
	{
		srcopt = true;
	}
	else
	{
		srcopt = false;
	}

	if (string(argv[7]) == "openmp")
	{
		codeOpt = string(argv[7]);
	}
	else if (string(argv[7]) == "gpu")
	{
		codeOpt = string(argv[7]);
	}
	else if (string(argv[7]) == "sequential")
	{
		codeOpt = string(argv[7]);
	}
	else
	{
		std::cout << "Invalid code optimization option." << endl;
		system("pause");
		exit(1);
	}

	if (string(argv[8]) == "3d" || string(argv[8]) == "3D" || string(argv[8]) == "threedim" || string(argv[8]) == "true" || string(argv[8]) == "yes")
	{
		threedim = true;
		if (test != "dam")
		{
			threedim = false;
			cout << endl << "The " << test << " test does not support 3D, a 2D simulation will be generated." << endl;
			system("pause");
		}
	}
	else
	{
		threedim = false;
	}

	if (string(argv[9]) == "outAll" || string(argv[9]) == "printall" || string(argv[9]) == "PRINTALL" || string(argv[9]) == "printAll" || string(argv[9]) == "true" || string(argv[9]) == "yes")
	{
		outAll = true;
	}
	else
	{
		outAll = false;
	}

	if (string(argv[10]) == "outFluid" || string(argv[10]) == "printfluid" || string(argv[10]) == "PRINTFLUID" || string(argv[10]) == "printFluid" || string(argv[10]) == "true" || string(argv[10]) == "yes")
	{
		outFluid = true;
	}
	else
	{
		outFluid = false;
	}


	int I = 0;

	double n0 = 0, Rho = 0.0;
	double* x, * y, * p, * u, * v, * SFX, * SFY;
	double* z, * w;


	int* PTYPE;
	double* xstar, * ystar, * ustar, * vstar, * pnew, * phat, * unew, * vnew, * NEUt, * TURB1, * TURB2, * TURB3, * n, * nstar, * MEU, * C, * RHO;
	double* zstar, * wstar, * wnew;


	int** neighb, *h_neighb;
	int* bcon;
	double** poiss, *h_poiss;
	double** ic, *h_ic;
	double* source;
	double dirichlet = 0.97;
	int imin = 10, imax = 50;
	double eps = 1.00e-9;
	double lambda;													//         MPS discretization coefficient
	double Xmin;													//         Minimum x of searching grid
	double Ymin;													//         Minimum y of searching grid
	double Zmin;													//         Minimum z of searching grid
	double Xmax;													//         Maximum x of searching grid
	double Ymax;													//         Maximum y of searching grid
	double Zmax;													//         Maximum z of searching grid
	int    FP;														//         Number of Fluid particles: in this model FP calculate in each time step
	int    WP;														//         Number of wall particles
	int    GP;														//         Number of ghost particles
	double c0 = 0.0;												//		   Speed of sound

	double Rho1 = stod(string(argv[11]));  //1000.0					//         Density of phase 1
	double NEU1 = stod(string(argv[12]));  //0.000001               //         Kinematic Viscosity 1
	double Rho2 = stod(string(argv[13]));  //500.0					//         Density of phase 2
	double NEU2 = stod(string(argv[14]));  //0.001                  //         Kinematic Viscosity 2

	double DL;
	if (!compressible) DL = 0.010;									//         Average particle distance (or particle size)  
	else DL = 0.0040;

	double re;
	if (threedim)
	{
		re = 3.2 * DL;												//		   Support area radius  
	}
	else
	{
		re = 4.0 * DL;
	}

	double BETA = 0.92;												//         Coefficient for determination of free surface
	double relaxp;
	if (!compressible) relaxp = 0.2;								//         Relaxation factor for pressure correction  
	else relaxp = 0.5;
	double COURANT = 1.0;											//         Courant number
	double correction = 0;											//         Correction factor to modify problem caused by shortage of ghost particles

	double coll;
	if (!compressible)
		coll = sqrt(DL * DL * 0.5 * 0.5);							//         Minimum particle distance to prevent collision  
	else
		coll = sqrt(DL * DL * 0.5 * 0.5);

	double velCor = 1.0;

	double CC;
	if (!compressible) CC = 0.20;									//         Collision coefficient
	else CC = 0.50;
	double MAXresi = 0.001;											//         Maximum acceptable residual for pressure calculation
	double Cs = 0.18;												//         Smogorinsky Constant (For using in SPS-LES turbulence model
	double DELTA = DL / 5.0;										//         Expansion value of background grid cells size.
	double Ncorrection = 1.0;										//         Correction of n0 value to improve the incompressibility and initial pressure fluctuation
	//double Ncorrection = 0.99;										//         Correction of n0 value to improve the incompressibility and initial pressure fluctuation
	int    SP = 0;													//         Number of storage particles
	int    TP;														//         Total number of particles
	int    KTYPE;													//         Kernel type
	if (!compressible)
		KTYPE = 2;
	else
		KTYPE = 6;


	int DIM;
	if (threedim)
	{
		DIM = 3;													//         Dimensions
	}
	else
	{
		DIM = 2;
	}

	int TURB;
	if (test == "turb")												//		   APPARENTLY TURBULENCE ENHANCES FLUID STABILITY OF CMPS-HS-HL-ECS 
		TURB = 1;													//         TURB=0: NO turbulence model, TURB=1: SPS turbulence model  
	else
	{
		if (turb)
			TURB = 1;
		else
			TURB = 0;
	}

	int Fraction_method = 2;										//         Method of calculation of volume of fraction. 1: Linear dist across the interface, 2: smoothed value

	double gx = 0.0;												//         Gravity acceleration in x direction
	double gy;														//         Gravity acceleration in y direction
	if (test == "turb")
	{
		gy = 0.0;
	}
	else
	{
		if (test == "shear")
			gy = 0.0;
		else
			gy = -9.8;
	}
	double gz = 0.0;
	double VMAX = 2.50;	 											//         To avoid jumping particles out of domain (If v>Vmax ---> v=Vmax)

	//--------------------Pressure and pressure gradient parameters--------------------

	int Method;
	if (!compressible)
		Method = 1;													//         Fully incompressible MPS: Method=1 ; Fully incompressible M-MPS: Method=2 ; Weakly compressible: Method=3 .  
	else
		Method = 3;


	double c01 = 20.0;												//         Numerical sound speed fluid 1. (Weakly compressible model)
	double c02 = 20.0;												//         Numerical sound speed fluid 2. (Weakly compressible model)
	int    KHcorrection = 1;										//         Khayyer and Gotoh pressure correction(1 = yes, 0 = no)
	int    IterMax = 100;											//         Maximum number of iterations for pressure calculation in each time step (Fully incompressible models)
	double PMAX;
	if (!compressible)
		PMAX = 10000.0;												//         A limit for the value of calculated pressure to avoid crashing  
	else
		PMAX = 20000.0;


	double PMIN = 0.0;												//         Minimum pressure, to avoid a high negative pressures
	bool loopstarted = false;

	//---------------------------Rheological parameters-----------------------------

	int    Fluid2_type;												//		   Newtonian:0  , H-B fluid:1 
	double N_PL = 0.0;													//		   flow behaviour (power law) index
	double MEU0 = 0;													//		   consistency index
	double PHI = 0;													//		   friction angle (RAD)
	double cohes = 0;													//		   cohesiveness coefficient
	double II = 0;
	double yield_stress = 0;
	if (visc)
	{
		Fluid2_type = 1;											//		   Newtonian:0  , H-B fluid:1 
		N_PL = 1.3;													//		   flow behaviour (power law) index
		MEU0 = 0.04;												//		   consistency index
		PHI = 0;													//		   friction angle (RAD)
		cohes = 0;													//		   cohesiveness coefficient
	}
	else
	{
		Fluid2_type = 0;											//		   Newtonian:0  , H-B fluid:1 
	}

	//------------------------------Time parameters-----------------------------------

	double  t, T = 2.0;												//         Simulation time (sec)
	T = stod(string(argv[15]));
	double* MAX = new double[1]();
	double* DT = new double[1]();
	if (stod(string(argv[16])) <= 0.005) DT[0] = stod(string(argv[16]));
	else
	{
		cout << "Time-step duration maximum value is 0.005 s (5 x 10^-3 s)" << endl;
		system("pause");
		exit(1);
	}
	double  DT_MAX = 0.005;											//         Maximum size of time interval allowed

	bool getFomVTU = false;											//		   Where to read input from


	cout << "==================================================================\n";
	cout << ".-.   .-..---.  .-.   .-.  .--.  ,---.        \n";
	cout << " \\ \\ / // .-. )  ) \\_/ /  / /\\ \\ | .-.\\ \n";
	cout << "  \\ V / | | |(_)(_)   /  / /__\\ \\| `-'/    \n";
	cout << "   ) /  | | | |   / _ \\  |  __  ||   (       \n";
	cout << "  (_)   \\ `-' /  / / ) \\ | |  |)|| |\\ \\   \n";
	cout << "         )---'  `-' (_)-'|_|  (_)|_| \\)\\    \n";
	cout << "        (_)       ,---.    .---.     (__)     \n";
	cout << "         |\\    /| | .-.\\  ( .-._)           \n";
	cout << "         |(\\  / | | |-' )(_) \\              \n";
	cout << "         (_)\\/  | | |--' _  \\ \\            \n";
	cout << "         | \\  / | | |   ( `-'  )             \n";
	cout << "         | |\\/| | /(     `----'              \n";
	cout << "         '-'  '-'(__)                         \n";
	cout << "\n";
	cout << " Based on 1. MPARS By: A. Shakibaeinia (2011)                 \n";
	cout << "          2. MPS method By: Koshizuka & Oka (1996)            \n";
	cout << "          3. CMPS-HS-HL-ECS method By: Khayyer & Gotoh (2013) \n";
	cout << "\n";
	cout << "\n";
	cout << "==================================================================\n";
	cout << "\n";
	cout << "\n";

	cout << "    Current execution details:\n";
	if (!compressible)
	{
		cout << "\tFULLY INCOMPRESSIBLE\n";
		if (matopt) cout << "\t    - HL (matrix optimization)\n";

		if (srcopt) cout << "\t    - HS & ECS (source term optimization)\n";
	}
	else
	{
		cout << "\tWEAKLY COMPRESSIBLE\n";
	}
	if (KHcorrection) cout << "\t    - CMPS (momentum enhancement)\n";
	cout << "\n";


	if (codeOpt == "gpu")
	{
		cout << "    EXECUTION: GPU CUDA execution\n";
	}
	else
	{
		if (codeOpt == "openmp")
		{
			cout << "    EXECUTION: CPU OpenMP execution\n";
		}
		else
		{
			cout << "    EXECUTION: CPU (sequential) execution\n";
		}
	}

	if (threedim)
		cout << "    DIMENSIONS: 3D\n";
	else
		cout << "    DIMENSIONS: 2D\n";

	if (TURB) cout << "    TURBULENCE: SPS Turbulence model\n";
	if (visc) cout << "    VISCOPLASTICITY: Herschel-Bulkley model (only for a second fluid in multiphase simulations)\n";

	if (threedim)
		cout << "    TEST: 3D DAM BREAK\n";
	else
	{
		if (test == "turb")
			cout << "    TEST: 2D POISEUILLE FLOW\n";
		else
		{
			if (test == "shear")
				cout << "    TEST: 2D SHEAR-DRIVEN CAVITY FLOW\n";
			else
			{
				if (test == "drop")
					cout << "    TEST: 2D WATER DROP\n";
				else
				{
					if (test == "oil")
						cout << "    TEST: 2D OIL SPILLING\n";
					else
						cout << "    TEST: 2D DAM BREAK\n";
				}
			}
		}
	}
	cout << "    TOTAL SIMULATION TIME: " << T << " seconds\n";
	cout << "    TIME-STEP DURATION: " << DT[0] << " seconds\n";

	cout << "\n";

	//-------------------------- defining LAMDA -----------------------------------------

	if (KTYPE == 1)   lambda = 0.22143 * pow(re, 2);
	if (KTYPE == 2)   lambda = 0.16832 * pow(re, 2);
	if (KTYPE == 5)   lambda = 0.250 * pow(re, 2);
	if (KTYPE == 6)   lambda = 0.14288 * pow(re, 2);

	ifstream tp;
	if (getFomVTU)
	{
		getTP(TP);
	}
	else
	{
		if (threedim)
		{
			tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-dam3d.grd");
		}
		else
		{
			if (test == "turb")
			{
				velCor = 0.8;
				tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-turb.grd");
			}
			else
			{
				if (test == "shear")
					tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-shear.grd");
				else
				{

					if (test == "drop")
					{
						tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-drop.grd");
					}
					else
					{
						if (test == "oil")
						{
							tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-oil.grd");
						}
						else
						{
							tp.open("..\\..\\..\\VoxarMPS\\inputs\\input-dam.grd");
						}
					}
				}
			}
		}
		if (!tp.is_open()) {
			cout << "Unable to read file." << endl;
			system("pause");
			exit(1);
		}
		tp >> TP;
		tp.close();
		tp.clear();
	}
	PMAX = (double)TP * 4.50;
	cout << "    Total Particle Number: " << TP << endl;
	cout << "    Maximum Pressure reached by one particle: " << PMAX << endl << endl << endl;

	//--------------------------- Defining Dynamic Matrices ------------------------------	
	x = new double[TP + 1]();
	y = new double[TP + 1]();
	/*if (threedim)*/ z = new double[TP + 1]();

	p = new double[TP + 1]();
	u = new double[TP + 1]();
	v = new double[TP + 1]();
	if (threedim) w = new double[TP + 1]();

	PTYPE = new int[TP + 1]();

	//---------------------- Openning INPUT and OUTPUT files ---------------------------------

	ifstream in;
	if (!getFomVTU)
	{

		if (threedim)
		{
			in.open("..\\..\\..\\VoxarMPS\\inputs\\input-dam3d.grd");
		}
		else
		{
			if (test == "turb")
			{
				velCor = 0.8;
				in.open("..\\..\\..\\VoxarMPS\\inputs\\input-turb.grd");
			}
			else
			{
				if (test == "shear")
					in.open("..\\..\\..\\VoxarMPS\\inputs\\input-shear.grd");
				else
				{

					if (test == "drop")
					{
						in.open("..\\..\\..\\VoxarMPS\\inputs\\input-drop.grd");
					}
					else
					{
						if (test == "oil")
						{
							in.open("..\\..\\..\\VoxarMPS\\inputs\\input-oil.grd");
						}
						else
						{
							in.open("..\\..\\..\\VoxarMPS\\inputs\\input-dam.grd");
						}
					}
				}
			}
		}
	}
	if(!in.is_open()){
		cout << "Unable to read file." << endl;
		system("pause");
		exit(1);
	}

	//----------------------- Reading Input file ---------------------------------------
	cout << "     READING INPUT FILE . . .\n";

	Xmin = 99999999;
	Ymin = 99999999;
	Zmin = 99999999;
	Xmax = -99999999;
	Ymax = -99999999;
	Zmax = -99999999;
	FP = 0;
	WP = 0;
	GP = 0;
	TP = 0;

	if (getFomVTU)
	{
		getVTU(x, y, z, u, v, w, p, PTYPE, TP, FP, GP, WP, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);
	}
	else
	{

		in >> TP;

		for (int I = 1; I <= TP; I++)
		{
			in >> x[I];
			in >> y[I];
			if (threedim) in >> z[I];

			in >> PTYPE[I];

			if (test == "drop")
			{
				in >> u[I];
				u[I] = -1 * x[I];
				in >> v[I];
				v[I] = 1 * y[I];
				if (threedim)
				{
					in >> w[I];
					w[I] = 0.0;
				}
			}
			else
			{
				in >> u[I];
				in >> v[I];
				if (threedim) in >> w[I];
			}

			in >> p[I];

			if (PTYPE[I] < 0)  GP++;
			if (PTYPE[I] == 0) WP++;
			if (PTYPE[I] > 0)  FP++;

			if (x[I] < Xmin) Xmin = x[I];
			if (x[I] > Xmax) Xmax = x[I];
			if (y[I] < Ymin) Ymin = y[I];
			if (y[I] > Ymax) Ymax = y[I];
			if (threedim)
			{
				if (z[I] < Zmin) Zmin = z[I];
				if (z[I] > Zmax) Zmax = z[I];
			}
		}
		in.close();
		in.clear();
	}

	TP = FP + GP + WP + SP;

	if (outAll)
		// saveParticles(TP, 0, x, y, z, u, v, w, p, PTYPE, DIM);

	if (outFluid)
		saveFluidParticles(TP, FP, 0, x, y, z, u, v, w, p, PTYPE, DIM);

	if (test == "drop")
	{
		Xmin = Xmin - 100 * DL;
		Ymin = Ymin - 10000 * DL;
		Xmax = Xmax + 100 * DL;
		Ymax = Ymax + 100 * DL;
		if (threedim)
		{
			Zmin = Zmin - DL;
			Zmax = Zmax + DL;
		}
	}
	else
	{
		Xmin = Xmin - DL;
		Ymin = Ymin - DL;
		Xmax = Xmax + DL;
		Ymax = Ymax + DL;
		if (threedim)
		{
			Zmin = Zmin - DL;
			Zmax = Zmax + DL;
		}
	}

	int ncx = int((Xmax - Xmin) / re) + 1;							//		   Number of cells in x direction
	int ncy = int((Ymax - Ymin) / re) + 1;							//		   Number of cells in y direction
	int ncz;
	if (threedim) ncz = int((Zmax - Zmin) / (re + DELTA)) + 1;      //		   Number of cells in z direction  

	int tnc;
	if (threedim) { tnc = ncx * ncy * ncz; }						//		   Total number of cells   
	else { tnc = ncx * ncy; }

	h_neighb = new int[(NEIGHBORS + 1) * (TP + 1)]();
	h_poiss = new double[(NEIGHBORS + 1) * (TP + 1)]();
	h_ic = new double[(NEIGHBORS + 1)*(TP + 1)]();
	neighb = new int* [TP + 1]();
	for (int m = 0; m <= TP; m++)
		neighb[m] = new int[NEIGHBORS + 1]();
	pnew = new double[TP + 1]();
	if (Method != 3)
	{
		bcon = new int[TP + 1]();
		source = new double[TP + 1]();
		poiss = new double* [TP + 1];
		for (int m = 1; m <= TP; m++)
			poiss[m] = new double[NEIGHBORS + 1]();
		ic = new double* [TP + 1];
		for (int m = 1; m <= TP; m++)
			ic[m] = new double[NEIGHBORS + 1]();
	}

	if (codeOpt != "gpu")
	{

		xstar = new double[TP + 1]();
		ystar = new double[TP + 1]();
		if (threedim) zstar = new double[TP + 1]();

		ustar = new double[TP + 1]();
		vstar = new double[TP + 1]();
		/*if (threedim)*/ wstar = new double[TP + 1]();

		phat = new double[TP + 1]();
		unew = new double[TP + 1]();
		vnew = new double[TP + 1]();
		/*if (threedim)*/ wnew = new double[TP + 1]();
		n = new double[TP + 1]();
		nstar = new double[TP + 1]();
		NEUt = new double[TP + 1]();
		TURB1 = new double[TP + 1]();
		TURB2 = new double[TP + 1]();
		if (threedim) TURB3 = new double[TP + 1]();

		C = new double[TP + 1]();
		MEU = new double[TP + 1]();
		RHO = new double[TP + 1]();
		SFX = new double[TP + 1]();
		SFY = new double[TP + 1]();




	}

	int* j = new int[1]();

	//----------------------- CUDA Settings --------------------------------

	//---Setting number of kernel calls for each case---
	int max_threads_per_block = 256;
	int division = TP / max_threads_per_block;
	int mod = TP % max_threads_per_block;

	double* d_MAX = NULL;
	double* d_DT = NULL;
	float* d_rqo = NULL;
	float* d_rqn = NULL;
	float* d_ps = NULL;
	double* d_aa = NULL;
	double* d_bb = NULL;
	double* d_err1 = NULL;
	int* d_j = NULL;
	double* d_x = NULL;
	double* d_y = NULL;
	double* d_z = NULL;
	double* d_p = NULL;
	double* d_u = NULL;
	double* d_v = NULL;
	double* d_w = NULL;
	int* d_PTYPE = NULL;
	double* d_nstar = NULL;
	double* d_xstar = NULL;
	double* d_ystar = NULL;
	double* d_zstar = NULL;
	double* d_ustar = NULL;
	double* d_vstar = NULL;
	double* d_wstar = NULL;
	double* d_pnew = NULL;
	double* d_phat = NULL;
	double* d_unew = NULL;
	double* d_vnew = NULL;
	double* d_wnew = NULL;
	double* d_NEUt = NULL;
	double* d_TURB1 = NULL;
	double* d_TURB2 = NULL;
	double* d_TURB3 = NULL;
	double* d_C = NULL;
	double* d_MEU = NULL;
	double* d_RHO = NULL;
	double* d_S11 = NULL;
	double* d_S12 = NULL;
	double* d_S22 = NULL;
	double* d_S13 = NULL;
	double* d_S23 = NULL;
	double* d_S33 = NULL;
	double* d_SFX = NULL;
	double* d_SFY = NULL;
	double* d_n = NULL;
	int* d_neighb = NULL;
	double* d_p1 = NULL;
	double* d_source = NULL;
	double* d_poiss = NULL;
	double* d_ic = NULL;
	int* d_ip = NULL;
	int* d_nc = NULL;
	double* d_q1 = NULL;
	double* d_r1 = NULL;
	double* d_s1 = NULL;
	double* d_aux = NULL;
	int* d_bcon = NULL;
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	const int Nrows = TP+1;
	const int Ncols = NEIGHBORS+1;

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

	int nnz = 0;                                // --- Number of nonzero elements in dense matrix
	const int lda = Nrows;                      // --- Leading dimension of dense matrix
	// --- Device side number of nonzero elements per row
	int* d_nnzPerVector = NULL;
	double* d_A = NULL;
	int* d_A_RowIndices = NULL;
	int* d_A_ColIndices = NULL;
	cusparseHandle_t cusparseHandle = 0;
	cublasHandle_t cublasHandle = 0;
	cusparseMatDescr_t descra = 0;
	cusparseMatDescr_t descrm = 0;
	cusparseSolveAnalysisInfo_t info_l = 0;
	cusparseSolveAnalysisInfo_t info_u = 0;
	double* devPtrF = 0;
	double* devPtrR = 0;
	double* devPtrRW = 0;
	double* devPtrPW = 0;
	double* devPtrS = 0;
	double* devPtrT = 0;
	double* devPtrV = 0;
	double ttt_sv = 0.0;
	int matrixM = TP+1;
	int matrixN = TP+1;
	double* devPtrMval = 0;
	int* devPtrMcolsIndex = 0;
	int* devPtrMrowsIndex = 0;
	const int max_iter = 1000;
	int k, M = 0, N = 0, nz = 0, * I_cal = NULL, * J_cal = NULL;
	int* d_col, * d_row;
	int qatest = 0;
	float * rhs;
	float r0, r1, alpha, beta;
	float* d_val;
	float* d_zm1, * d_zm2, * d_rm2;
	float* d_r, * d_p_cal, * d_omega, * d_y_cal;
	float* val = NULL;
	float* d_valsILU0;
	float* valsILU0;
	float rsum, diff, err = 0.0;
	float qaerr1, qaerr2 = 0.0;
	float dot, numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	
	int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP, arraySizePW, arraySizeS, arraySizeT, arraySizeV, mNNZ;

	/* compressed sparse row */
	arraySizeX = matrixN;
	arraySizeF = matrixM;
	arraySizeR = matrixM;
	arraySizeRW = matrixM;
	arraySizeP = matrixN;
	arraySizePW = matrixN;
	arraySizeS = matrixM;
	arraySizeT = matrixM;
	arraySizeV = matrixM;
	cusparseStatus_t status1, status2, status3;

	/* initialize cublas */
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	/* initialize cusparse */
	status1 = cusparseCreate(&cusparseHandle);
	if (status1 != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUSPARSE initialization error\n");
		return EXIT_FAILURE;
	}
	/* create three matrix descriptors */
	status1 = cusparseCreateMatDescr(&descra);
	status2 = cusparseCreateMatDescr(&descrm);
	if ((status1 != CUSPARSE_STATUS_SUCCESS) ||
		(status2 != CUSPARSE_STATUS_SUCCESS)) {
		fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
		return EXIT_FAILURE;
	}
	int maxit = 2000; //5; //2000; //1000;  //50; //5; //50; //100; //500; //10000;
	double tol = 0.0000001; //0.000001; //0.00001; //0.00000001; //0.0001; //0.001; //0.00000001; //0.1; //0.001; //0.00000001;
	int *A_RowIndices = new int[ (Nrows + 1) * sizeof(*d_A_RowIndices)];

	if (codeOpt == "gpu")
	{

		/* allocate device memory for csr matrix and vectors */
		checkCudaErrors(cudaMalloc((void**)&devPtrF, sizeof(devPtrF[0]) * arraySizeF));
		checkCudaErrors(cudaMalloc((void**)&devPtrR, sizeof(devPtrR[0]) * arraySizeR));
		checkCudaErrors(cudaMalloc((void**)&devPtrRW, sizeof(devPtrRW[0]) * arraySizeRW));
		checkCudaErrors(cudaMalloc((void**)&devPtrPW, sizeof(devPtrPW[0]) * arraySizePW));
		checkCudaErrors(cudaMalloc((void**)&devPtrS, sizeof(devPtrS[0]) * arraySizeS));
		checkCudaErrors(cudaMalloc((void**)&devPtrT, sizeof(devPtrT[0]) * arraySizeT));
		checkCudaErrors(cudaMalloc((void**)&devPtrV, sizeof(devPtrV[0]) * arraySizeV));
		/*checkCudaErrors(cudaMemcpy(A_RowIndices, d_A_RowIndices, (Nrows) * sizeof(int), cudaMemcpyDeviceToHost));
		int mNNZ = A_RowIndices[matrixM + 1] - A_RowIndices[1];
		int mSizeAval = mNNZ;*/
		checkCudaErrors(cudaMalloc((void**)&devPtrMval, sizeof(devPtrMval[0]) * Nrows));
		checkCudaErrors(cudaMemset((void*)devPtrMval, 0, sizeof(devPtrMval[0]) * Nrows));

		//-------Allocating arrays in device memory-------
		gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
		gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
		gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));
		gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
		/* clean memory */
		checkCudaErrors(cudaMemset((void*)devPtrF, 0, sizeof(devPtrF[0]) * arraySizeF));
		checkCudaErrors(cudaMemset((void*)devPtrR, 0, sizeof(devPtrR[0]) * arraySizeR));
		checkCudaErrors(cudaMemset((void*)devPtrRW, 0, sizeof(devPtrRW[0]) * arraySizeRW));
		checkCudaErrors(cudaMemset((void*)devPtrPW, 0, sizeof(devPtrPW[0]) * arraySizePW));
		checkCudaErrors(cudaMemset((void*)devPtrS, 0, sizeof(devPtrS[0]) * arraySizeS));
		checkCudaErrors(cudaMemset((void*)devPtrT, 0, sizeof(devPtrT[0]) * arraySizeT));
		checkCudaErrors(cudaMemset((void*)devPtrV, 0, sizeof(devPtrV[0]) * arraySizeV));

		

		cudaMalloc((void**)& d_MAX, (1) * sizeof(double));

		cudaMalloc((void**)& d_DT, (1) * sizeof(double));
		cudaMalloc((void**)& d_rqo, (1) * sizeof(float));
		cudaMalloc((void**)& d_rqn, (1) * sizeof(float));
		cudaMalloc((void**)& d_ps, (1) * sizeof(float));
		cudaMalloc((void**)& d_aa, (1) * sizeof(double));
		cudaMalloc((void**)& d_bb, (1) * sizeof(double));
		cudaMalloc((void**)& d_err1, (1) * sizeof(double));
		cudaMalloc((void**)& d_j, (1) * sizeof(int));

		cudaMalloc((void**)& d_x, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_y, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_z, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_p, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_u, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_v, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_w, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_PTYPE, (TP + 1) * sizeof(int));
		cudaMalloc((void**)& d_nstar, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_xstar, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_ystar, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_zstar, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_ustar, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_vstar, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_wstar, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_pnew, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_phat, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_unew, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_vnew, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_wnew, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_NEUt, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_TURB1, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_TURB2, (TP + 1) * sizeof(double));
		if (threedim) cudaMalloc((void**)& d_TURB3, (TP + 1) * sizeof(double));

		cudaMalloc((void**)& d_C, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_MEU, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_RHO, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_S11, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_S12, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_S22, (TP + 1) * sizeof(double));

		if (threedim)
		{
			cudaMalloc((void**)& d_S13, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_S23, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_S33, (TP + 1) * sizeof(double));
		}

		cudaMalloc((void**)& d_n, (TP + 1) * sizeof(double));
		cudaMalloc((void**)& d_neighb, (NEIGHBORS + 1) * (TP + 1) * sizeof(int));
		cudaMemset(d_neighb, 0, (NEIGHBORS + 1) * (TP + 1) * sizeof(int));

		if (!compressible)
		{
			cudaMalloc((void**)& d_p1, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_q1, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_r1, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_s1, (TP + 1) * sizeof(double));
			cudaMalloc((void**)& d_aux, (TP + 1) * sizeof(double));

			cudaMalloc((void**)& d_bcon, (TP + 1) * sizeof(int));

			cudaMalloc((void**)& d_source, (TP + 1) * sizeof(double));

			cudaMalloc((void**)& d_poiss, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));
			cudaMemset(d_poiss, 0, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));

			cudaMalloc((void**)& d_ic, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));
			cudaMemset(d_ic, 0, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));
		}

		cudaMalloc((void**)& d_ip, (TP + 1) * sizeof(int));
	}

	int divisionNeigh1, divisionEuler, divisionPress;
	int modNeigh1, modEuler, modPress;

	int* d_Ista, * d_Iend;

	if (codeOpt == "gpu")
	{
		divisionNeigh1 = (tnc) / max_threads_per_block;
		modNeigh1 = (tnc)-(divisionNeigh1 * max_threads_per_block);

		divisionEuler = (TP - (GP + WP)) / max_threads_per_block;
		modEuler = (TP - (GP + WP)) - (divisionEuler * max_threads_per_block);

		divisionPress = (TP - GP) / max_threads_per_block;
		modPress = (TP - GP) - (divisionPress * max_threads_per_block);

		d_Ista = NULL;
		d_Iend = NULL;
		cudaMalloc((void**)& d_Ista, (tnc + 1) * sizeof(int));
		cudaMalloc((void**)& d_Iend, (tnc + 1) * sizeof(int));
	}

	cout << "     PRE- ITERATION CALCULATIONS... \n";

	if (codeOpt == "gpu")
	{
		//---------------------------------- CUDA memory transfers and sets -----------------------------------------

		cudaMemcpy(d_x, x, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_u, u, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_v, v, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_p, p, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_PTYPE, PTYPE, (TP + 1) * sizeof(int), cudaMemcpyHostToDevice);
		if (threedim)
		{
			cudaMemcpy(d_w, w, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_z, z, (TP + 1) * sizeof(double), cudaMemcpyHostToDevice);
		}
	}

	//------------------------------------------ Density assigning ---------------------------------------------------

	if (codeOpt == "gpu")
	{
		if (division > 0) setDensity << <division, max_threads_per_block >> > (1, d_RHO, TP, d_PTYPE, Rho1, Rho2);
		if (mod > 0) setDensity << <1, mod >> > ((division * max_threads_per_block) + 1, d_RHO, TP, d_PTYPE, Rho1, Rho2);
	}
	else
	{
		for (int I = 1; I <= TP; I++)
		{
			if (PTYPE[I] == 2)RHO[I] = Rho2;
			else            RHO[I] = Rho1;
		}
	}

	//------------------------------------------ Neighborhood calculation -------------------------------------------

	int Cnum;
	int* Ista, * Iend;
	int icell, jcell, kcell;
	int* ip = NULL;

	if (codeOpt != "gpu")
	{
		//NEIGHBOR(Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, re, DELTA, TP, x, y, z, neighb, DIM, codeOpt);
		neighbour_cuda_2d(TP, x, y, DELTA, re, neighb, Xmin, Xmax, Ymin, Ymax);
	}
	else
	{
		neighbour_cuda_2d_gpu(TP, d_x, d_y, DELTA, re, d_neighb, Xmin, Xmax, Ymin, Ymax);
	}
	

	//--------------------------- Calculation of initial particle number density ---------------------------------------

	if (codeOpt != "gpu")
	{
		PNUM(I, KTYPE, DIM, TP, re, neighb, x, y, z, Ncorrection, n, MAX, pnew, p, loopstarted, codeOpt);
	}
	else
	{
		if (division > 0) pnum << <division, max_threads_per_block >> > (1, KTYPE, DIM, TP, re, d_neighb, d_x, d_y, d_z, Ncorrection, d_n, d_MAX, d_pnew, d_p, loopstarted);
		if (mod > 0) pnum << <1, mod >> > ((division * max_threads_per_block) + 1, KTYPE, DIM, TP, re, d_neighb, d_x, d_y, d_z, Ncorrection, d_n, d_MAX, d_pnew, d_p, loopstarted);

		cudaMemcpy(MAX, d_MAX, (1) * sizeof(double), cudaMemcpyDeviceToHost);
	}
	n0 = MAX[0];
	cout << "Particle number density: " << n0 << endl;

	// --------------------------- Calculation of CPU time --------------------------------
	auto start1 = chrono::high_resolution_clock::now();

	//-------------------------------------------------------------------------------------
	//------------------------------- Time Iteration --------------------------------------
	//-------------------------------------------------------------------------------------
	cout << "\n";
	cout << "     ITERATION STARTED  (each bar represents 1 time step) \n";
	t = 0;

	for (int Tstep = 1; t <= T; Tstep++)
	{
		bool loopstarted = true;
		DTcalculation(c0, c01, c02, DT, DT_MAX, COURANT, DL);

		TP = FP + WP + GP;

		//--------------------- Calculation of particle number density ---------------------

		if (Method != 3)
		{
			if (codeOpt != "gpu")
			{
				PNUM(I, KTYPE, DIM, TP, re, neighb, x, y, z, Ncorrection, n, MAX, pnew, p, loopstarted, codeOpt);
			}
			else
			{
				if (division > 0) pnum << <division, max_threads_per_block >> > (1, KTYPE, DIM, TP, re, d_neighb, d_x, d_y, d_z, Ncorrection, d_n, d_MAX, d_pnew, d_p, loopstarted);
				if (mod > 0) pnum << <1, mod >> > ((division * max_threads_per_block) + 1, KTYPE, DIM, TP, re, d_neighb, d_x, d_y, d_z, Ncorrection, d_n, d_MAX, d_pnew, d_p, loopstarted);
			}
		}
		//----------------------------- Neighborhood calculation --------------------------

		if (((Tstep - 1) / stepsToCalcNeigh) == int((Tstep - 1) / stepsToCalcNeigh))
		{
			if (codeOpt != "gpu")
			{
				//NEIGHBOR(Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, re, DELTA, TP, x, y, z, neighb, DIM, codeOpt);
				neighbour_cuda_2d(TP, x, y, DELTA, re, neighb, Xmin, Xmax, Ymin, Ymax);
			}
			else
			{
				neighbour_cuda_2d_gpu(TP, d_x, d_y, DELTA, re, d_neighb, Xmin, Xmax, Ymin, Ymax);
			}
		}

		//----------------------------- SPS-LES tubulence calculation --------------------------

		if (TURB == 1)
		{
			if (codeOpt != "gpu")
			{
				SPS(re, TP, GP, neighb, x, y, z, coll, KTYPE, unew, vnew, wnew, n0, NEUt, Cs, DL, TURB1, TURB2, TURB3, DIM, codeOpt);
			}
			else
			{
				if (divisionPress > 0) turb1 << <divisionPress, max_threads_per_block >> > (GP + 1, re, d_neighb, d_x, d_y, d_z, coll, KTYPE, d_unew, d_vnew, d_wnew, n0, d_NEUt, Cs, DL, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);
				if (modPress > 0) turb1 << <1, modPress >> > ((divisionPress * max_threads_per_block) + GP + 1, re, d_neighb, d_x, d_y, d_z, coll, KTYPE, d_unew, d_vnew, d_wnew, n0, d_NEUt, Cs, DL, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);

				if (divisionPress > 0) turb2 << <divisionPress, max_threads_per_block >> > (GP + 1, re, d_neighb, d_x, d_y, d_z, coll, KTYPE, n0, d_NEUt, d_TURB1, d_TURB2, d_TURB3, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);
				if (modPress > 0) turb2 << <1, modPress >> > ((divisionPress * max_threads_per_block) + GP + 1, re, d_neighb, d_x, d_y, d_z, coll, KTYPE, n0, d_NEUt, d_TURB1, d_TURB2, d_TURB3, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);
			}
		}

		//----------------------------- Volume fraction calculation --------------------------

		if (codeOpt != "gpu")
		{
			V_FRACTION(re, Fraction_method, TP, neighb, PTYPE, C, KTYPE, x, y, z, DIM, codeOpt);
		}
		else
		{
			if (division > 0) volFraction << <division, max_threads_per_block >> > (1, re, Fraction_method, TP, d_neighb, d_PTYPE, d_C, KTYPE, d_x, d_y, d_z, DIM);
			if (mod > 0) volFraction << <1, mod >> > ((division * max_threads_per_block) + 1, re, Fraction_method, TP, d_neighb, d_PTYPE, d_C, KTYPE, d_x, d_y, d_z, DIM);
		}

		//----------------------------- Calculation of dynamic viscosity --------------------------

		if (codeOpt != "gpu")
		{
			VISCOSITY(re, TP, Fluid2_type, PTYPE, MEU, NEU1, NEU2, Rho1, Rho2, neighb, x, y, z, KTYPE, unew, vnew, wnew, n0, C, PHI, I, cohes, II, yield_stress, phat, MEU0, N_PL, DIM, codeOpt);
		}
		else
		{
			if (division > 0) viscosity << <division, max_threads_per_block >> > (1, re, TP, Fluid2_type, d_PTYPE, d_MEU, NEU1, NEU2, Rho1, Rho2, d_neighb, d_x, d_y, d_z, KTYPE, d_unew, d_vnew, d_wnew, n0, d_C, PHI, I, cohes, II, yield_stress, d_phat, MEU0, N_PL, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);
			if (mod > 0) viscosity << <1, mod >> > ((division * max_threads_per_block) + 1, re, TP, Fluid2_type, d_PTYPE, d_MEU, NEU1, NEU2, Rho1, Rho2, d_neighb, d_x, d_y, d_z, KTYPE, d_unew, d_vnew, d_wnew, n0, d_C, PHI, I, cohes, II, yield_stress, d_phat, MEU0, N_PL, d_S11, d_S12, d_S22, d_S13, d_S23, d_S33, DIM);
		}

		//------------------------------------ Prediction --------------------------------------

		if (codeOpt != "gpu")
		{
			PREDICTION(re, xstar, ystar, zstar, ustar, vstar, wstar, u, v, w, TP, PTYPE, MEU, Rho1, Rho2, neighb, x, y, z, KTYPE, n0, phat, pnew, gx, gy, gz, DT[0], NEUt, lambda, TURB1, TURB2, TURB3, relaxp, RHO, SFX, SFY, DIM, codeOpt);

		}
		else
		{
			if (division > 0) prediction << <division, max_threads_per_block >> > (1, re, d_xstar, d_ystar, d_zstar, d_ustar, d_vstar, d_wstar, d_u, d_v, d_w, TP, d_PTYPE, d_MEU, Rho1, Rho2, d_neighb, d_x, d_y, d_z, KTYPE, n0, d_phat, d_pnew, gx, gy, gz, DT[0], d_NEUt, lambda, d_TURB1, d_TURB2, d_TURB3, relaxp, d_RHO, d_SFX, d_SFY, DIM);
			if (mod > 0) prediction << <1, mod >> > ((division * max_threads_per_block) + 1, re, d_xstar, d_ystar, d_zstar, d_ustar, d_vstar, d_wstar, d_u, d_v, d_w, TP, d_PTYPE, d_MEU, Rho1, Rho2, d_neighb, d_x, d_y, d_z, KTYPE, n0, d_phat, d_pnew, gx, gy, gz, DT[0], d_NEUt, lambda, d_TURB1, d_TURB2, d_TURB3, relaxp, d_RHO, d_SFX, d_SFY, DIM);
		}

		//----------------------------- Particle number density calculation ------------------------------

		if (codeOpt != "gpu")
		{
			PNUMSTAR(KTYPE, DIM, TP, re, neighb, xstar, ystar, zstar, nstar, codeOpt);
		}
		else
		{
			if (division > 0) pnumstar << <division, max_threads_per_block >> > (1, KTYPE, DIM, TP, re, d_neighb, d_xstar, d_ystar, d_zstar, d_nstar);
			if (mod > 0) pnumstar << <1, mod >> > ((division * max_threads_per_block) + 1, KTYPE, DIM, TP, re, d_neighb, d_xstar, d_ystar, d_zstar, d_nstar);
		}

		//--------------------------- Boundary condition for matrix calculation --------------------------------

		if (Method != 3)
		{
			if (codeOpt != "gpu")
			{
				BCON(PTYPE, bcon, nstar, n0, dirichlet, TP, codeOpt);
			}
			else
			{
				if (division > 0) bconCalc << <division, max_threads_per_block >> > (1, d_PTYPE, d_bcon, d_nstar, n0, dirichlet, TP);
				if (mod > 0) bconCalc << <1, mod >> > ((division * max_threads_per_block) + 1, d_PTYPE, d_bcon, d_nstar, n0, dirichlet, TP);
			}
		}
		//------------------------------ Pressure calculation ------------------------------------

		if (codeOpt != "gpu")
		{
			PRESSURECALC(Method, GP, FP, WP, TP, PTYPE, c0, c01, c02, Rho1, Rho2, C, nstar, BETA, n0, pnew, PMIN, PMAX, IterMax, MAXresi, re,
				x, y, z, coll, KTYPE, correction, Rho, relaxp, lambda, DT[0], p, n, DIM, neighb, poiss, bcon, source, ic, imin, imax, eps, ustar, vstar, wstar, matopt, srcopt, codeOpt);
		}
		else
		{
			if (Method == 3)
			{
				if (divisionPress > 0) pressureCalcWC << <divisionPress, max_threads_per_block >> > (GP + 1, d_PTYPE, c0, c01, c02, Rho1, Rho2, d_C, d_nstar, BETA, n0, d_pnew, PMIN, PMAX, Rho);
				if (modPress > 0) pressureCalcWC << <1, modPress >> > ((divisionPress * max_threads_per_block) + GP + 1, d_PTYPE, c0, c01, c02, Rho1, Rho2, d_C, d_nstar, BETA, n0, d_pnew, PMIN, PMAX, Rho);
			}
			else if (Method == 1)
			{
				cudaMemset(d_poiss, 0, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));
				cudaMemset(d_ic, 0, (NEIGHBORS + 1) * (TP + 1) * sizeof(double));
				cudaMemset(d_source, 0, (TP + 1) * sizeof(double));

				//cudaMemcpy(h_neighb, d_neighb, (NEIGHBORS + 1)* (TP+1) * sizeof(int), cudaMemcpyDeviceToHost);
				std::ofstream outneigh;
				std::string filename = "outneigh_convert" + std::to_string(Tstep) + ".txt";
				outneigh.open(filename, 'w');

				if (division > 0) matrixCalc << <division, max_threads_per_block >> > (1, re, d_xstar, d_ystar, d_zstar, KTYPE, d_nstar, n0, Rho1, lambda, DT[0], DIM, d_neighb, d_poiss, d_bcon, matopt);
				if (mod > 0) matrixCalc << <1, mod >> > ((division * max_threads_per_block) + 1, re, d_xstar, d_ystar, d_zstar, KTYPE, d_nstar, n0, Rho1, lambda, DT[0], DIM, d_neighb, d_poiss, d_bcon, matopt);

				if (division > 0) sourceCalc << <division, max_threads_per_block >> > (1, TP, d_PTYPE, d_bcon, d_nstar, n0, DT[0], d_source, d_neighb, d_xstar, d_ystar, d_zstar, d_ustar, d_vstar, d_wstar, re, DIM, srcopt);
				if (mod > 0) sourceCalc << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_PTYPE, d_bcon, d_nstar, n0, DT[0], d_source, d_neighb, d_xstar, d_ystar, d_zstar, d_ustar, d_vstar, d_wstar, re, DIM, srcopt);

				if (division > 0) incdecom << <division, max_threads_per_block >> > (1, TP, d_bcon, d_neighb, d_poiss, d_ic);
				if (mod > 0) incdecom << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_bcon, d_neighb, d_poiss, d_ic);
				cudaDeviceSynchronize();
				cudaError_t error = cudaGetLastError();

				//*********************************************************************************************************************************************************************************************************************

				//cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_poiss, lda, d_nnzPerVector, &nnz));
				//cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_poiss, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));
				//checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
				//checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));

				//int matrixSizeAval = nnz;
				///* analyse the lower and upper triangular factors */
				//double ttl = second();
				//checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
				//checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
				//checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, d_A, d_A_RowIndices, d_A_ColIndices, info_l));
				//checkCudaErrors(cudaDeviceSynchronize());
				//double ttl2 = second();

				//double ttu = second();
				//checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
				//checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
				//checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, d_A, d_A_RowIndices, d_A_ColIndices, info_u));
				//checkCudaErrors(cudaDeviceSynchronize());
				//double ttu2 = second();
				//ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
				//printf("analysis lower %f (s), upper %f (s) \n", ttl2 - ttl, ttu2 - ttu);

				//checkCudaErrors(cudaMemcpy(devPtrMval, d_A, (size_t)(matrixSizeAval * sizeof(devPtrMval[0])), cudaMemcpyDeviceToDevice));
				///* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
				//double start_ilu, stop_ilu;
				//printf("CUSPARSE csrilu0 ");
				//start_ilu = second();
				//devPtrMrowsIndex = d_A_RowIndices;
				//devPtrMcolsIndex = d_A_ColIndices;

				//checkCudaErrors(cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, descra, devPtrMval, d_A_RowIndices, d_A_ColIndices, info_l));
				//checkCudaErrors(cudaDeviceSynchronize());
				//stop_ilu = second();
				//fprintf(stdout, "time(s) = %10.8f \n", stop_ilu - start_ilu);

				///* run the test */
				//int num_iterations = 1; //10; 
				//for (int count = 0; count < num_iterations; count++) {

				//	gpu_pbicgstab(cublasHandle, cusparseHandle, matrixM, matrixN, nnz,
				//		descra, d_A, d_A_RowIndices, d_A_ColIndices,
				//		descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex,
				//		info_l, info_u,
				//		devPtrF, devPtrR, devPtrRW, d_source, devPtrPW, devPtrS, devPtrT, devPtrV, d_pnew, maxit, tol, ttt_sv);

				//}

				///* destroy the analysis info (for lower and upper triangular factors) */
				//checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
				//checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));


				//*********************************************************************************************************************************************************************************************************************

				//to host: bcon, neighb, poiss, source, ic

				//cudaMemcpy(h_neighb, d_neighb, (NEIGHBORS + 1)* (TP+1) * sizeof(int), cudaMemcpyDeviceToHost);

				//for (int j = 1; j <= TP; j++) {
				//	for (int i = 0; i <= h_neighb[j * (NEIGHBORS)+1]; i++) {
				//		neighb[j][i + 2] = h_neighb[j * (NEIGHBORS)+i + 2];
				//	}
				//	neighb[j][1] = h_neighb[j * (NEIGHBORS) + 1];
				//}

				//cudaMemcpy(h_poiss, d_poiss, (NEIGHBORS + 1)* (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);

				//for (int j = 1; j <= TP; j++) {
				//	for (int i = 0; i < NEIGHBORS; i++) {
				//		poiss[j][i + 1] = h_poiss[j * (NEIGHBORS)+i + 1];
				//	}
				//	//poiss[j][1] = h_poiss[j * (NEIGHBORS)+1];
				//}


				//cudaMemcpy(h_ic, d_ic, (NEIGHBORS + 1)* (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);

				//for (int j = 1; j <= TP; j++) {
				//	for (int i = 0; i < NEIGHBORS; i++) {
				//		ic[j][i + 1] = h_ic[j * (NEIGHBORS)+i + 1];
				//	}
				//	//ic[j + 1][1] = h_ic[j * (NEIGHBORS)+1];
				//}


				////for (int j = 1; j <= TP; j++) {
				////	for (int i = 0; i < NEIGHBORS; i++) {
				////		outneigh << poiss[j][i+1] << " ";
				////	}
				////	outneigh << std::endl;
				////}

				////outneigh.close();
				//cudaMemcpy(bcon, d_bcon, (TP + 1) * sizeof(int), cudaMemcpyDeviceToHost);
				//cudaMemcpy(source, d_source, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				//cudaMemset(d_pnew, 0, (TP + 1) * sizeof(double));
				//cudaDeviceSynchronize();

				//CGM(TP, source, IterMax, MAXresi, poiss, neighb, bcon, pnew, imax, ic, DT[0], eps, imin, codeOpt);


				//for (int I = 1; I <= TP; I++)
				//{
				//	if (pnew[I] < PMIN)
				//	{
				//		pnew[I] = PMIN;
				//	}
				//	if (pnew[I] > PMAX || pnew[I] * 0.0 != 0.0)
				//	{
				//		if (pnew[I] * 0.0 != 0.0) cout << "pressao dando infinito...\n";
				//		pnew[I] = PMAX;
				//	}

				//}

				////to device: pnew
				//cudaMemcpy(d_pnew, pnew, sizeof(double) * (TP + 1), cudaMemcpyHostToDevice);

				
				if (division > 0) cgm1 << <division, max_threads_per_block >> > (1, TP, d_bcon, d_s1, d_neighb, d_poiss, d_pnew);
				if (mod > 0) cgm1 << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_bcon, d_s1, d_neighb, d_poiss, d_pnew);

				if (division > 0) cgm2 << <division, max_threads_per_block >> > (1, d_r1, d_source, d_s1);
				if (mod > 0) cgm2 << <1, mod >> > ((division * max_threads_per_block) + 1, d_r1, d_source, d_s1);

				cudaMemcpy(d_aux, d_r1, sizeof(double) * (TP + 1), cudaMemcpyDeviceToDevice);

				if (division > 0) cgmFS << <division, max_threads_per_block >> > (1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
				if (mod > 0) cgmFS << <1, mod >> > ((division * max_threads_per_block) + 1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);

				cudaMemcpy(d_aux, d_q1, sizeof(double) * (TP + 1), cudaMemcpyDeviceToDevice);

				if (division > 0) cgmBS << <division, max_threads_per_block >> > (1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
				if (mod > 0) cgmBS << <1, mod >> > ((division * max_threads_per_block) + 1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);

				cudaMemcpy(d_p1, d_q1, sizeof(double) * (TP + 1), cudaMemcpyDeviceToDevice);
				cudaMemset(d_rqo, 0, sizeof(float));

				if (division > 0) cgm5 << <division, max_threads_per_block >> > (1, d_r1, d_q1, d_bcon, d_rqo);
				if (mod > 0) cgm5 << <1, mod >> > ((division * max_threads_per_block) + 1, d_r1, d_q1, d_bcon, d_rqo);

				for (int k = 0; k < imax; k++)
				{

					if (division > 0) cgm6 << <division, max_threads_per_block >> > (1, TP, d_bcon, d_s1, d_neighb, d_poiss, d_p1);
					if (mod > 0) cgm6 << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_bcon, d_s1, d_neighb, d_poiss, d_p1);

					cudaMemset(d_ps, 0, sizeof(float));

					if (division > 0) cgm7 << <division, max_threads_per_block >> > (1, d_p1, d_s1, d_bcon, d_ps);
					if (mod > 0) cgm7 << <1, mod >> > ((division * max_threads_per_block) + 1, d_p1, d_s1, d_bcon, d_ps);

					aaKernel << <1, 1 >> > (d_aa, d_rqo, d_ps);

					if (division > 0) cgm8 << <division, max_threads_per_block >> > (1, d_pnew, d_p1, d_aa);
					if (mod > 0) cgm8 << <1, mod >> > ((division * max_threads_per_block) + 1, d_pnew, d_p1, d_aa);

					if (division > 0) cgm9 << <division, max_threads_per_block >> > (1, d_r1, d_s1, d_aa);
					if (mod > 0) cgm9 << <1, mod >> > ((division * max_threads_per_block) + 1, d_r1, d_s1, d_aa);

					cudaMemcpy(d_aux, d_r1, sizeof(double) * (TP + 1), cudaMemcpyDeviceToDevice);

					if (division > 0) cgmFS << <division, max_threads_per_block >> > (1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
					if (mod > 0) cgmFS << <1, mod >> > ((division * max_threads_per_block) + 1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
					
					cudaMemcpy(d_aux, d_q1, sizeof(double) * (TP + 1), cudaMemcpyDeviceToDevice);

					if (division > 0) cgmBS << <division, max_threads_per_block >> > (1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
					if (mod > 0) cgmBS << <1, mod >> > ((division * max_threads_per_block) + 1, d_ic, d_q1, d_aux, d_bcon, d_neighb, TP);
					
					cudaMemset(d_rqn, 0, sizeof(float));

					if (division > 0) cgm10 << <division, max_threads_per_block >> > (1, d_r1, d_q1, d_bcon, d_rqn);
					if (mod > 0) cgm10 << <1, mod >> > ((division * max_threads_per_block) + 1, d_r1, d_q1, d_bcon, d_rqn);

					bbKernel << <1, 1 >> > (d_bb, d_rqn, d_rqo);

					cudaMemcpy(d_rqo, d_rqn, sizeof(float), cudaMemcpyDeviceToDevice);

					if (division > 0) cgm11 << <division, max_threads_per_block >> > (1, d_p1, d_q1, d_bb);
					if (mod > 0) cgm11 << <1, mod >> > ((division * max_threads_per_block) + 1, d_p1, d_q1, d_bb);

					cudaMemset(d_j, 0, sizeof(int));

					if (division > 0)  cgm12 << <division, max_threads_per_block >> > (1, d_r1, d_bcon, d_err1, d_j, eps, DT[0]);
					if (mod > 0)  cgm12 << <1, mod >> > ((division * max_threads_per_block) + 1, d_r1, d_bcon, d_err1, d_j, eps, DT[0]);

					cudaMemcpy(j, d_j, 1*sizeof(int), cudaMemcpyDeviceToHost);


					if (j[0] == 0 && k >= imin) break;
				}

				if (division > 0) pnewSet << <division, max_threads_per_block >> > (1, d_pnew, PMIN, PMAX);
				if (mod > 0) pnewSet << <1, mod >> > ((division * max_threads_per_block) + 1, d_pnew, PMIN, PMAX);

			}
		}


		//------------------------------- Calculation of Phat ------------------------------------

		if (codeOpt != "gpu")
		{
			PHATCALC(TP, neighb, pnew, phat, codeOpt);
		}
		else
		{
			if (division > 0) phatCalc << <division, max_threads_per_block >> > (1, TP, d_neighb, d_pnew, d_phat);
			if (mod > 0) phatCalc << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_neighb, d_pnew, d_phat);
		}

		//------------------------ Calculation of pressure gradient ----------------------------

		if (codeOpt != "gpu")
		{
			PRESSGRAD(GP, WP, KHcorrection, TP, pnew, neighb, xstar, ystar, zstar, phat, KTYPE, re, RHO, ustar, vstar, wstar, DT[0], unew, vnew, wnew, relaxp, n0, VMAX, DIM, codeOpt, nstar);
		}
		else
		{
			if (divisionEuler > 0) pressGrad << <divisionEuler, max_threads_per_block >> > ((GP + WP + 1), GP, WP, KHcorrection, TP, d_pnew, d_neighb, d_xstar, d_ystar, d_zstar, d_phat, KTYPE, re, d_RHO, d_ustar, d_vstar, d_wstar, DT[0], d_unew, d_vnew, d_wnew, relaxp, n0, VMAX, DIM);
			if (modEuler > 0) pressGrad << <1, modEuler >> > ((divisionEuler * max_threads_per_block) + (GP + WP + 1), GP, WP, KHcorrection, TP, d_pnew, d_neighb, d_xstar, d_ystar, d_zstar, d_phat, KTYPE, re, d_RHO, d_ustar, d_vstar, d_wstar, DT[0], d_unew, d_vnew, d_wnew, relaxp, n0, VMAX, DIM);
		}

		//---------------------------------- Moving particles -----------------------------------

		if (codeOpt != "gpu")
		{
			EULERINTEGRATION(GP, WP, TP, DT[0], x, y, z, unew, vnew, wnew, DL, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, I, velCor, DIM, test, codeOpt);
		}
		else
		{
			if (divisionEuler > 0) eulerIntegration << <divisionEuler, max_threads_per_block >> > ((GP + WP + 1), GP, WP, TP, DT[0], d_x, d_y, d_z, d_unew, d_vnew, d_wnew, DL, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, I, CC, DIM);
			if (modEuler > 0) eulerIntegration << <1, modEuler >> > ((divisionEuler * max_threads_per_block) + (GP + WP + 1), GP, WP, TP, DT[0], d_x, d_y, d_z, d_unew, d_vnew, d_wnew, DL, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, I, CC, DIM);
		}

		//------------------------------- Aplying the pair-wise Collision -------------------------------

		if (codeOpt != "gpu")
		{
			COLLISION2(TP, coll, PTYPE, Rho1, Rho2, neighb, CC, unew, vnew, wnew, x, y, z, DT[0], DIM, codeOpt);
		}
		else
		{
			if (division > 0) collision2 << <division, max_threads_per_block >> > (1, TP, coll, d_PTYPE, Rho1, Rho2, d_neighb, CC, d_unew, d_vnew, d_wnew, d_x, d_y, d_z, DT[0], DIM);
			if (mod > 0) collision2 << <1, mod >> > ((division * max_threads_per_block) + 1, TP, coll, d_PTYPE, Rho1, Rho2, d_neighb, CC, d_unew, d_vnew, d_wnew, d_x, d_y, d_z, DT[0], DIM);
		}

		//------------------------------ Prepare data for new time step --------------------------------

		if (codeOpt != "gpu")
		{
			PREPDATA(TP, FP, x, y, z, u, v, w, p, unew, vnew, wnew, pnew, Xmin, Ymin, Xmax, Ymax, Zmin, Zmax, DIM, test, codeOpt);
		}
		else
		{
			if (division > 0) prepData << <division, max_threads_per_block >> > (1, TP, d_x, d_y, d_z, d_u, d_v, d_w, d_p, d_unew, d_vnew, d_wnew, d_pnew, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, DIM);
			if (mod > 0) prepData << <1, mod >> > ((division * max_threads_per_block) + 1, TP, d_x, d_y, d_z, d_u, d_v, d_w, d_p, d_unew, d_vnew, d_wnew, d_pnew, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, DIM);

			if (outAll || outFluid)
			{
				cudaMemcpy(x, d_x, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(y, d_y, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(p, d_p, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(u, d_u, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(v, d_v, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				if (threedim)
				{
					cudaMemcpy(z, d_z, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(w, d_w, (TP + 1) * sizeof(double), cudaMemcpyDeviceToHost);
				}
			}
		}

		//------------------------------------------ Printing results --------------------------------------------------------

		if (outAll)
			saveParticles(TP, Tstep, x, y, /*z,*/ u, v, /*w,*/ p, PTYPE, DIM);


		if (outFluid)
			saveFluidParticles(TP, FP, Tstep, x, y, z, u, v, w, p, PTYPE, DIM);


		t = t + DT[0];

		if (Tstep == 1)
		{
			auto stop1 = chrono::high_resolution_clock::now();
			auto duration = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
			auto duration_hours = chrono::duration_cast<chrono::hours>(stop1 - start1);

			cout << "\n     Estimated FPS: " << 1000.0 / duration.count() << " (Frames Per Second)\n" << endl;
			cout << "     Estimated running time: " << T / DT[0] * duration.count() / 60000.0 << " minutes (Total running time)\n" << endl;
			cout << "                          OR " << T / DT[0] * duration.count() / 3600000.0 << " hours (Total running time)\n" << endl;
			cout << "                          OR " << duration.count() << " msec (per time step)\n" << endl;
		}
		cout << "|";
	}
	// --------------------------------End of Time Loop --------------------------------------------------

	cout << "End!\n";
	return 0;
}

