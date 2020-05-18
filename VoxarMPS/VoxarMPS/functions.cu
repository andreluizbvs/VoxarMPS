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
#include <map>
#include <tuple>
#include <vector>
#include <algorithm>
using namespace std;


//===========================================================================================
//=====================     Kernel function       ===========================================
//===========================================================================================

double W(double R, int KTYPE, int dim, double re)
{
	double w;
	double q = R / re;
	switch (KTYPE) {
		case 1: 											//		   Second order polynomial function (Koshizuka and Oka, 1996)
			if (q < 0.5) w = 2.0 - 4.0 * pow(q, 2);
			else if (q <= 1.0) w = (2 * q - 2) * (2 * q - 2);
			else w = 0;
			break;

		case 2:												//		   Rational function (Koshizuka et al., 1998)
			if (q < 1.0) w = (1 / q) - 1;
			else w = 0;
			break;

		case 3:												//		   Cubic spline function
			double C;
			if (dim == 1) C = 0.6666;
			if (dim == 2) C = 1.43 * 3.14;
			if (dim == 3) C = 1 / 3.14;

			if (R < re) w = C / pow(re, dim) * (1 - 1.5 * pow(q, 2.0) + 0.75 * pow(q, 3.0));
			else if (R >= re && R < (2.0 * re)) w = C / pow(re, dim) * (0.25 * pow(2.0 - q, 3.0));
			else w = 0;
			break;

		case 5:												//		   Cubic spline function
			if (q <= 1.0) w = 1.5 * log(1 / (q + 0.000001));
			else w = 0;
			break;

		case 6:												//		   3rd order polynomial function (Shakibaeinia and Jin, 2010)
			if (q <= 1.0) w = pow((1 - q), 3);
			else w = 0;
			break;
	}
	return (w);
}

double diff(int i, int j, double* x)
{
	return (x[j] - x[i]);
}

double dist2d(int i, int j, double* x, double* y)
{
	double R;
	R = sqrt(pow(diff(i, j, x), 2.0) + pow(diff(i, j, y), 2.0));
	return (R);
}

double dist3d(int i, int j, double* x, double* y, double* z)
{
	double R;
	R = sqrt(pow(diff(i, j, x), 2.0) + pow(diff(i, j, y), 2.0) + pow(diff(i, j, z), 2.0));
	return (R);
}

//=================================================================================================
//=====================     Particle Number Density Calc.  ========================================
//=================================================================================================
double* PNUM(int i, int ktype, int dim, int TP, double re, int** neighb, double* x, double* y, double* z, double Ncorrection, double* n, double* MAX, double* pnew, double* p, bool loopstarted, string codeOpt)
{

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		double sum = 0.0;
		double d = 0;
		for (int l2 = 2; l2 <= neighb[I][1]; l2++)
		{
			int J = neighb[I][l2];
			if (dim == 3)
				d = dist3d(I, J, x, y, z);
			else
				d = dist2d(I, J, x, y);

			if (I != J) sum = sum + W(d, ktype, dim, re);
		}
		if (ktype != 2)sum = sum * Ncorrection;
		n[I] = sum;
		if (!loopstarted)
		{
			pnew[I] = p[I];
			if (n[I] > MAX[0])   MAX[0] = n[I];
		}
	}
	return(n);
}

__global__ void pnum(int offset, int ktype, int dim, int TP, double re, int* neighb, double* x, double* y, double* z, double Ncorrection, double* n, double* MAX, double* pnew, double* p, bool loopstarted)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);


	double sum = 0.0;
	double d = 0;

	for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
	{
		int J = neighb[I * NEIGHBORS + l];
		if (dim == 3)
			d = dist3d(I, J, x, y, z);
		else
			d = dist2d(I, J, x, y);

		if (I != J) sum = sum + W(d, ktype, dim, re);
	}
	if (ktype != 2) sum = sum * Ncorrection;
	n[I] = sum;
	if (!loopstarted)
	{
		pnew[I] = p[I];
		if (n[I] > MAX[0])   MAX[0] = n[I];
	}

}


//=================================================================================================
//=====================     Particle Number Density * Calc.  ======================================
//=================================================================================================
double* PNUMSTAR(int ktype, int dim, int TP, double re, int** neighb, double* xstar, double* ystar, double* zstar, double* nstar, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		double sum = 0.0;
		double d = 0;
		int J;
		for (int l = 2; l <= neighb[I][1]; l++)
		{
			J = neighb[I][l];
			if (dim == 3)
				d = dist3d(I, J, xstar, ystar, zstar);
			else
				d = dist2d(I, J, xstar, ystar);

			if (I != J) sum = sum + W(d, ktype, dim, re);
		}
		nstar[I] = sum;
	}
	return(nstar);
}

__global__ void pnumstar(int offset, int KTYPE, int DIM, int TP, double re, int* d_neighb, double* d_xstar, double* d_ystar, double* d_zstar, double* d_nstar)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double sum = 0.0;
	double d = 0.0;
	int J = 0;
	for (int l = 2; l <= d_neighb[I * NEIGHBORS + 1]; l++)
	{
		J = d_neighb[I * NEIGHBORS + l];
		if (DIM == 3)
			d = sqrt(pow(d_xstar[J] - d_xstar[I], 2.0) + pow(d_ystar[J] - d_ystar[I], 2.0) + pow(d_zstar[J] - d_zstar[I], 2.0));
		else
			d = sqrt(pow(d_xstar[J] - d_xstar[I], 2.0) + pow(d_ystar[J] - d_ystar[I], 2.0));

		if (I != J) sum = sum + W(d, KTYPE, DIM, re);
	}

	d_nstar[I] = sum;
}


//=================================================================================================
//================================   Particle hat Calc.  ==========================================
//=================================================================================================
double* PHATCALC(int TP, int** neighb, double* pnew, double* phat, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		double min = 999999999999999;
		for (int l = 2; l <= neighb[I][1]; l++)
		{
			int J = neighb[I][l];

			if (pnew[J] < min)min = pnew[J];
		}
		phat[I] = min;
	}
	return phat;
}

__global__ void phatCalc(int offset, int TP, int* neighb, double* pnew, double* phat)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double min = 999999999999999;
	for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
	{
		int J = neighb[I * NEIGHBORS + l];

		if (pnew[J] < min)min = pnew[J];
	}
	phat[I] = min;

}


//=================================================================================================
//===========================     Pressure gradient Calc.  ========================================
//=================================================================================================
void PRESSGRAD(int GP, int WP, int KHcorrection, int TP, double* pnew, int** neighb, double* xstar, double* ystar, double* zstar, double* phat, int KTYPE, double re, double* RHO, double* ustar, double* vstar, double* wstar, double DT,
	double* unew, double* vnew, double* wnew, double relaxp, double n0, double VMAX, int dim, string codeOpt, double *nstar)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = GP + WP + 1; I <= TP; I++)
	{

		double sum1 = 0;
		double sum2 = 0;
		double sum3 = 0;

		double summat1 = 0;
		double summat2 = 0;
		double summat3 = 0;
		double vij = 1.0 / nstar[I];
		//if (KHcorrection == 1)
		if (true)
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];
				double D;
				if (dim == 3)
					D = dist3d(I, J, xstar, ystar, zstar);
				else
					D = dist2d(I, J, xstar, ystar);

				if (I != J)
				{
					sum1 = sum1 + (pnew[J] + pnew[I] - (phat[I] + phat[J])) * diff(I, J, xstar) * W(D, KTYPE, dim, re) / D / D;
					sum2 = sum2 + (pnew[J] + pnew[I] - (phat[I] + phat[J])) * diff(I, J, ystar) * W(D, KTYPE, dim, re) / D / D;
					if (dim == 3)
						sum3 = sum3 + (pnew[J] + pnew[I] - (phat[I] + phat[J])) * (zstar[J] - zstar[I]) * W(D, KTYPE, dim, re) / D / D;

					summat1 = summat1 + diff(I, J, xstar) * diff(I, J, xstar) * vij * W(D, KTYPE, dim, re) / D / D;
					summat2 = summat2 + diff(I, J, xstar) * diff(I, J, ystar) * vij * W(D, KTYPE, dim, re) / D / D;
					summat3 = summat3 + diff(I, J, ystar) * diff(I, J, ystar) * vij * W(D, KTYPE, dim, re) / D / D;

				}
			}
		}
		
		double det = 1.0 / (summat1 * summat3 - summat2 * summat2);
		if (KHcorrection == 2) {
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];
				double D;
				if (dim == 3)
					D = dist3d(I, J, xstar, ystar, zstar);
				else
					D = dist2d(I, J, xstar, ystar);

				if (I != J)
				{
					sum1 = sum1 + (pnew[J] - pnew[I]) / D / D * diff(I, J, xstar) * W(D, KTYPE, dim, re);
					sum2 = sum2 + (pnew[J] - pnew[I]) / D / D * diff(I, J, ystar) * W(D, KTYPE, dim, re) ;
					if (dim == 3)
						sum3 = sum3 + (pnew[J] - pnew[I]) / D / D * diff(I, J, zstar) * W(D, KTYPE, dim, re);


				}
			}
		}
		else if(KHcorrection == 0)
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];
				double D;
				if (dim == 3)
					D = dist3d(I, J, xstar, ystar, zstar);
				else
					D = dist2d(I, J, xstar, ystar);

				if (I != J)
				{

					sum1 = sum1 + (pnew[J] - phat[I]) * diff(I, J, xstar) * W(D, KTYPE, dim, re) / D / D;
					sum2 = sum2 + (pnew[J] - phat[I]) * diff(I, J, ystar) * W(D, KTYPE, dim, re) / D / D;
					if (dim == 3)
						sum3 = sum3 + (pnew[J] - phat[I]) * (zstar[J] - zstar[I]) * W(D, KTYPE, dim, re) / D / D;


				}
			}
		}
		double Rho = RHO[I];
		unew[I] = ustar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum1;
		vnew[I] = vstar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum2;
		if (dim == 3)
			wnew[I] = wstar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum3;

		//----------- Damper ----------------------------				

		if (fabs(unew[I]) > 2.0 * VMAX) unew[I] = VMAX;
		if (vnew[I] > 2.0 * VMAX) vnew[I] = 2.0 * VMAX;
		if (vnew[I] < -2.0 * VMAX) vnew[I] = -2.0 * VMAX;
		if (dim == 3)
			if (fabs(wnew[I]) > 2.0 * VMAX) wnew[I] = VMAX;
		//------------------------------------------------

	}
}


__global__ void pressGrad(int offset, int GP, int WP, int KHcorrection, int TP, double* pnew, int* neighb, double* xstar, double* ystar, double* zstar, double* phat, int KTYPE, double re,
	double* RHO, double* ustar, double* vstar, double* wstar, double DT, double* unew, double* vnew, double* wnew, double relaxp, double n0, double VMAX, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double sum1 = 0;
	double sum2 = 0;
	double sum3 = 0;

	if (KHcorrection == 1)
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			int J = neighb[I * NEIGHBORS + l];
			double D;
			if (dim == 3)
				D = sqrt(pow(xstar[J] - xstar[I], 2.0) + pow(ystar[J] - ystar[I], 2.0) + pow(zstar[J] - zstar[I], 2.0));
			else
				D = sqrt(pow(xstar[J] - xstar[I], 2.0) + pow(ystar[J] - ystar[I], 2.0));

			double w = W(D, KTYPE, dim, re);

			if (I != J)
			{
				sum1 = sum1 + (pnew[J] + pnew[I] - 2.0 * phat[I]) * (xstar[J] - xstar[I]) * w / D / D;
				sum2 = sum2 + (pnew[J] + pnew[I] - 2.0 * phat[I]) * (ystar[J] - ystar[I]) * w / D / D;
				if (dim == 3)
					sum3 = sum3 + (pnew[J] + pnew[I] - 2.0 * phat[I]) * (zstar[J] - zstar[I]) * w / D / D;
			}
		}
	}
	else
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			int J = neighb[I * NEIGHBORS + l];
			double D;
			if (dim == 3)
				D = sqrt(pow(xstar[J] - xstar[I], 2.0) + pow(ystar[J] - ystar[I], 2.0) + pow(zstar[J] - zstar[I], 2.0));
			else
				D = sqrt(pow(xstar[J] - xstar[I], 2.0) + pow(ystar[J] - ystar[I], 2.0));

			double w = W(D, KTYPE, dim, re);

			if (I != J)
			{

				sum1 = sum1 + (pnew[J] - phat[I]) * (xstar[J] - xstar[I]) * w / D / D;
				sum2 = sum2 + (pnew[J] - phat[I]) * (ystar[J] - ystar[I]) * w / D / D;
				if (dim == 3)
					sum3 = sum3 + (pnew[J] - phat[I]) * (zstar[J] - zstar[I]) * w / D / D;

			}
		}
	}
	double Rho = RHO[I];
	unew[I] = ustar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum1;
	vnew[I] = vstar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum2;
	if (dim == 3)
		wnew[I] = wstar[I] - relaxp * (2.0 * DT / n0 / Rho) * sum3;

	//----------- Damper ----------------------------				

	if (fabs(unew[I]) > 2.0 * VMAX) unew[I] = VMAX;
	if (vnew[I] > 2.0 * VMAX) vnew[I] = 2.0 * VMAX;
	if (vnew[I] < -2.0 * VMAX) vnew[I] = -2.0 * VMAX;
	if (dim == 3)
		if (fabs(wnew[I]) > 2.0 * VMAX) wnew[I] = VMAX;

	//------------------------------------------------
}

//===========================================================================================
//====================  Boundary condition of matrix calculation ============================
//===========================================================================================
int* BCON(int* PTYPE, int* bcon, double* n, double n0, double dirichlet, int TP, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (PTYPE[i] <= -1) bcon[i] = -1;
		else if (n[i] / n0 < dirichlet) bcon[i] = 1;
		else bcon[i] = 0;
	}
	return bcon;
}

__global__ void bconCalc(int offset, int* PTYPE, int* bcon, double* n, double n0, double dirichlet, int TP)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (PTYPE[i] <= -1) bcon[i] = -1;
	else if (n[i] / n0 < dirichlet) bcon[i] = 1;
	else bcon[i] = 0;
}



//========================================================================================
//====================================  Check bcon =======================================
//========================================================================================
int* CHECKBCON(int* check, int* bcon, int** neigh, int TP, string codeOpt)
{
	int /*i, j, l,*/ count;

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (bcon[i] == -1) check[i] = -1;
		else if (bcon[i] == 1) check[i] = 1;
		else check[i] = 0;
	}

	do
	{
		count = 0;
		for (int i = 1; i <= TP; i++)
		{
			if (check[i] == 1)
			{
				for (int l = 2; l <= neigh[i][1]; l++)
				{
					int j = neigh[i][l];
					if (check[j] == 0)check[j] = 1;
				}
				check[i] = 2;
				count++;
			}
		}
	} while (count != 0);

	for (int i = 1; i <= TP; i++)
	{
		if (check[i] == 0) cout << "Warning no Dirichlet boundary, i = " << i << endl;
	}
	return check;
}


//=================================================================================================
//==============   Conjugate Gradient Method For Linear Eq.s   ====================================
//=================================================================================================


void CGM(int TP, double* b, int IterMax, double MAXresi, double** poiss, int** neigh, int* bcon, double* x, int imax, double** ic, double dt, double eps, int imin, string codeOpt)
{
	double* r = new double[(TP + 1)]();
	double* p = new double[(TP + 1)]();
	double* q = new double[(TP + 1)]();
	double* s = new double[(TP + 1)]();
	double* aux = new double[(TP + 1)]();

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (bcon[i] != 0) continue;
		s[i] = poiss[i][1] * x[i];
		for (int l = 2; l <= neigh[i][1]; l++)
		{
			int j = neigh[i][l];
			if (bcon[j] == -1) continue;
			s[i] = s[i] + poiss[i][l] * x[j];
		}
	}

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)r[i] = b[i] - s[i];

	memcpy(aux, r, sizeof(double) * (TP + 1));

	/*  forward substitution */
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (bcon[i] != 0) { continue; }
		for (int l = 2; l <= neigh[i][1]; l++)
		{
			int j = neigh[i][l];
			if (j > i) { continue; }
			if (bcon[j] != 0) { continue; }
			aux[i] = aux[i] - ic[i][l] * q[j];
		}
		q[i] = aux[i] / ic[i][1];
	}

	memcpy(aux, q, sizeof(double) * (TP + 1));

	/*  backward substitution */
	if (codeOpt == "openmp") {
#pragma omp parallel for schedule (guided)
	}
	for (int i = TP; i >= 1; i--)
	{
		if (bcon[i] != 0) { continue; }
		for (int l = 2; l <= neigh[i][1]; l++)
		{
			int j = neigh[i][l];
			if (j < i) { continue; }
			if (bcon[j] != 0) { continue; }
			aux[i] = aux[i] - ic[i][l] * q[j];
		}
		q[i] = aux[i] / ic[i][1];
	}

	memcpy(p, q, sizeof(double) * (TP + 1));
	double rqo = 0.0;
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (bcon[i] == 0)
		{
			rqo += r[i] * q[i];
		}
	}

	for (int k = 0; k < imax; k++)
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (bcon[i] != 0) continue;
			s[i] = poiss[i][1] * p[i];
			for (int l = 2; l <= neigh[i][1]; l++)
			{
				int j = neigh[i][l];
				if (bcon[j] == -1) continue;
				s[i] = s[i] + poiss[i][l] * p[j];
			}
		}

		double ps = 0.0;

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (bcon[i] == 0)
			{
				ps += p[i] * s[i];
			}
		}

		double aa = rqo / ps;

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			x[i] = x[i] + aa * p[i];
		}

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)r[i] = r[i] - aa * s[i];

		//--------------------------------------------------
		//BEGIN SOLVER
		memcpy(aux, r, sizeof(double) * (TP + 1));

		/*  forward substitution */
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (bcon[i] != 0) continue;
			for (int l = 2; l <= neigh[i][1]; l++)
			{
				int j = neigh[i][l];
				if (j > i) continue;
				if (bcon[j] != 0) continue;
				aux[i] = aux[i] - ic[i][l] * q[j];
			}
			q[i] = aux[i] / ic[i][1];
		}

		memcpy(aux, q, sizeof(double) * (TP + 1));

		/*  backward substitution */
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = TP; i >= 1; i--)
		{
			if (bcon[i] != 0) continue;
			for (int l = 2; l <= neigh[i][1]; l++)
			{
				int j = neigh[i][l];
				if (j < i) continue;
				if (bcon[j] != 0) continue;
				aux[i] = aux[i] - ic[i][l] * q[j];
			}
			q[i] = aux[i] / ic[i][1];
		}

		double rqn = 0.0;
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (bcon[i] == 0)
			{
				rqn += r[i] * q[i];
			}
		}

		double bb = rqn / rqo;
		rqo = rqn;

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)p[i] = q[i] + bb * p[i];

		int j = 0;
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (bcon[i] != 0) continue;
			double err1;

			if (r[i] > 0)
			{
				err1 = r[i] * dt * dt;
			}
			else
			{
				err1 = -r[i] * dt * dt;
			}
			if (err1 > eps)
			{
				j++;
			}
		}

		if (j == 0 && k >= imin)
		{
#ifndef _DEBUG
			delete[] p;
			delete[] r;
			delete[] q;
			delete[] s;
			delete[] aux;
#endif
			r = NULL;
			p = NULL;
			q = NULL;
			s = NULL;
			aux = NULL;
			break;
		}
	}
#ifndef _DEBUG
	delete[] r;
	delete[] p;
	delete[] q;
	delete[] s;
	delete[] aux;
#endif
	r = NULL;
	p = NULL;
	q = NULL;
	s = NULL;
	aux = NULL;
}

__global__ void cgm1(int offset, int TP, int* bcon, double* s, int* neigh, double* poiss, double* x)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		s[i] = poiss[i * NEIGHBORS + 1] * x[i];
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (bcon[j] == -1)continue;
			s[i] = s[i] + poiss[i * NEIGHBORS + l] * x[j];
		}
	}
}

__global__ void cgm2(int offset, double* r, double* b, double* s)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	r[i] = b[i] - s[i];
}

__global__ void cgmFS(int offset, double* ic, double* q, double* aux, int* bcon, int* neigh, int TP)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (j > i) continue;
			if (bcon[j] != 0) continue;
			aux[i] = aux[i] - ic[i * NEIGHBORS + l] * q[j];
		}
		q[i] = aux[i] / ic[i * NEIGHBORS + 1];
	}

	//for (int i = 1; i <= TP; i++)
	//{
	//	if (bcon[i] == 0)
	//	{
	//		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
	//		{
	//			int j = neigh[i * NEIGHBORS + l];
	//			if (j > i) continue;
	//			if (bcon[j] != 0) continue;
	//			aux[i] = aux[i] - ic[i * NEIGHBORS + l] * q[j];
	//		}
	//		q[i] = aux[i] / ic[i * NEIGHBORS + 1];
	//	}
	//}
}

__global__ void cgmBS(int offset, double* ic, double* q, double* aux, int* bcon, int* neigh, int TP)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (j < i)continue;
			if (bcon[j] != 0)continue;
			aux[i] = aux[i] - ic[i * NEIGHBORS + l] * q[j];
		}
		q[i] = aux[i] / ic[i * NEIGHBORS + 1];
	}

	//for (int i = TP; i >= 1; i--)
	//{
	//	if (bcon[i] == 0)
	//	{
	//		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
	//		{
	//			int j = neigh[i * NEIGHBORS + l];
	//			if (j < i)continue;
	//			if (bcon[j] != 0)continue;
	//			aux[i] = aux[i] - ic[i * NEIGHBORS + l] * q[j];
	//		}
	//		q[i] = aux[i] / ic[i * NEIGHBORS + 1];
	//	}
	//}
}

__global__ void cgm5(int offset, double* r, double* q, int* bcon, float* rqo)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		float addInc = r[i] * q[i];
		atomicAdd(&rqo[0], addInc);
	}
}

__global__ void cgm6(int offset, int TP, int* bcon, double* s, int* neigh, double* poiss, double* p)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		s[i] = poiss[i * NEIGHBORS + 1] * p[i];
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (bcon[j] == -1)continue;
			s[i] = s[i] + poiss[i * NEIGHBORS + l] * p[j];
		}
	}
}

__global__ void cgm7(int offset, double* p, double* s, int* bcon, float* ps)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		float addInc = p[i] * s[i];
		atomicAdd(&ps[0], addInc);
	}
}

__global__ void cgm8(int offset, double* x, double* p, double* aa)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	x[i] = x[i] + aa[0] * p[i];
}

__global__ void cgm9(int offset, double* r, double* s, double* aa)
{

	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	r[i] = r[i] - aa[0] * s[i];
}

__global__ void cgm10(int offset, double* r, double* q, int* bcon, float* rqn)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		float addInc = r[i] * q[i];
		atomicAdd(&rqn[0], addInc);
	}
}

__global__ void cgm11(int offset, double* p, double* q, double* bb)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	p[i] = q[i] + bb[0] * p[i];
}

__global__ void cgm12(int offset, double* r, int* bcon, double* err1, int* j, double eps, double dt)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double err1_d;
	if (bcon[i] == 0)
	{
		if (r[i] > 0) { err1_d = r[i] * dt * dt; }
		else { err1_d = -r[i] * dt * dt; }
		if (err1_d > eps)
		{
			atomicAdd(&j[0], 1);
		}
	}
}


//=================================================================================================
//=======================  Preparing MATRIXES (A and b for x) for CGM  ============================
//=================================================================================================

void MATRIX(double re, int Method, int FP, int WP, int TP, double* x, double* y, double* z, double coll, int KTYPE, int* PTYPE, double correction,
	double* nstar, double BETA, double n0, double* pnew, double Rho, double relaxp, double lambda, double DT,
	double* p, double* n, int GP, int dim, int** neigh, double** poiss, int* bcon, double* source, double* unew, double* vnew, double* wnew, bool matopt, bool srcopt, string codeOpt)
{
	if (Method == 1)
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			double val = 1.0;

			if (bcon[i] != 0) continue;
			poiss[i][1] = 0.0;
			for (int l = 2; l <= neigh[i][1]; l++)
			{
				int j = neigh[i][l];
				if (i == j) continue;

				if (bcon[j] == -1)poiss[i][l] = 0.0;
				else
				{
					val = 1.0;
					double d;
					if (dim == 3)
						d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0) + pow((z[j] - z[i]), 2.0));
					else
						d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0));

					if (fabs(d) <= 1.0e-15) continue;

					if (matopt)
					{
						if (dim == 2)
						{
							val = 3.0 * re;
						}
						else
						{
							val = 2.0 * re;
						}
						val = val / (d * d * d);
						val = val / n0;
						val = val / Rho;
					}
					else
					{
						val = 2.0 * dim / lambda * W(d, KTYPE, dim, re) / n0 / Rho;
					}

					poiss[i][l] = -val;
					poiss[i][1] += val;
				}
			}

			poiss[i][1] += (1.00e-7) / DT / DT;
		}

		if (srcopt)
		{
			if (codeOpt == "openmp")
			{
#pragma omp parallel for schedule (guided)
			}
			for (int i = 1; i <= TP; i++)
			{
				if (PTYPE[i] == -1) continue;
				if (bcon[i] == 0)
				{
					double alpha = 0.0;
					double beta = 0.0;
					double ECS = 0.0;
					if (i > 1)
					{
						alpha = (nstar[i] - n0) / n0; // (n[i]-n0p)/n0p
						beta = (DT / n0) * (source[i - 1] / (-1.0 / (n0 * DT))); // (dt/n0)*(Dn/Dt)
						ECS = fabs(alpha) * (beta / DT) + fabs(beta) * (alpha / DT);
					}
					double sum = 0.0;

					for (int l = 2; l <= neigh[i][1]; l++)
					{
						int j = neigh[i][l];

						if (PTYPE[j] <= -1 || i == j) continue;

						double d;
						if (dim == 3)
						{
							d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0) + pow((z[j] - z[i]), 2.0));
						}
						else
						{
							d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0));
						}

						if (fabs(d) <= 1.0e-15) continue;

						double du = unew[i] - unew[j];
						double dv = vnew[i] - vnew[j];
						double dw;
						if (dim == 3)
						{
							dw = wnew[i] - wnew[j];
							sum += (re / (d * d * d)) * ((x[i] - x[j]) * du + (y[i] - y[j]) * dv + (z[i] - z[j]) * dw);
						}
						else
							sum += (re / (d * d * d)) * ((x[i] - x[j]) * du + (y[i] - y[j]) * dv);

					}
					source[i] = (-1.25 / (n0 * DT)) * sum + ECS;
					if (fabs(source[i]) < 0.99) source[i] = 0;

				}
				else if (bcon[i] == 1)
				{
					source[i] = 0.0;
				}
			}
		}
		else
		{
			if (codeOpt == "openmp")
			{
#pragma omp parallel for schedule (guided)
			}
			for (int i = 1; i <= TP; i++)
			{
				if (PTYPE[i] == -1) continue;
				if (bcon[i] == 0)source[i] = 1.0 / DT / DT * (n[i] - n0) / n0;
				else if (bcon[i] == 1)
				{
					source[i] = 0.0;
				}
			}
		}
	}
}

__global__ void matrixCalc(int offset, double re, double* x, double* y, double* z, int KTYPE, double* nstar, double n0, double Rho, double lambda, double DT, int dim, int* neigh, double* poiss, int* bcon, bool matopt)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);
	double val = 1.0;

	if (bcon[i] == 0)
	{

		poiss[i * NEIGHBORS + 1] = 0.0;
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (i == j) continue;
			if (bcon[j] == -1)poiss[i * NEIGHBORS + l] = 0.0;
			else
			{
				val = 1.0;
				double d;
				if (dim == 3)
					d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0) + pow((z[j] - z[i]), 2.0));
				else
					d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0));

				if (fabs(d) <= 1.0e-15) continue;

				if (matopt)
				{
					if (dim == 2)
					{
						val = 3.0 * re;
					}
					else
					{
						val = 2.0 * re;
					}
					val = val / (d * d * d);
					val = val / n0;
					val = val / Rho;
				}
				else
				{
					val = 2.0 * dim / lambda * W(d, KTYPE, dim, re) / n0 / Rho;
				}

				poiss[i * NEIGHBORS + l] = -val;
				poiss[i * NEIGHBORS + 1] += val;
			}
		}

		poiss[i * NEIGHBORS + 1] += (1.00e-7) / DT / DT;

	}

}

__global__ void sourceCalc(int offset, int TP, int* PTYPE, int* bcon, double* nstar, double n0, double DT, double* source, int* neigh, double* x, double* y, double* z,
	double* unew, double* vnew, double* wnew, double re, int dim, bool srcopt)
{

	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (srcopt)
	{
		if (PTYPE[i] != -1)
		{
			if (bcon[i] == 0)
			{
				double alpha = 0.0;
				double beta = 0.0;
				double ECS = 0.0;
				if (i > 1)
				{
					alpha = (nstar[i] - n0) / n0; // (n[i]-n0p)/n0p
					beta = (DT / n0) * (source[i - 1] / (-1.0 / (n0 * DT))); // (dt/n0)*(Dn/Dt)
					ECS = fabs(alpha) * (beta / DT) + fabs(beta) * (alpha / DT);
				}
				double sum = 0.0;

				for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
				{
					int j = neigh[i * NEIGHBORS + l];

					if (PTYPE[j] <= -1 || i == j) continue;
					double d;
					if (dim == 3) {
						d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0) + pow((z[j] - z[i]), 2.0));
					}
					else {
						d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0));
					}

					if (fabs(d) <= 1.0e-15)	continue;

					double du = unew[i] - unew[j];
					double dv = vnew[i] - vnew[j];
					if (dim == 3)
					{
						double dw = wnew[i] - wnew[j];
						sum += (re / (d * d * d)) * ((x[i] - x[j]) * du + (y[i] - y[j]) * dv + (z[i] - z[j]) * dw); /////////////// MPS-HS
					}
					else
						sum += (re / (d * d * d)) * ((x[i] - x[j]) * du + (y[i] - y[j]) * dv); /////////////// MPS-HS

				}
				source[i] = (-1.25 / (n0 * DT)) * sum + ECS;
				if (fabs(source[i]) < 0.99) source[i] = 0;

			}
			else if (bcon[i] == 1)
			{
				source[i] = 0.0;
			}
		}

	}
	else
	{
		if (PTYPE[i] != -1)
		{
			if (bcon[i] == 0)source[i] = 1.0 / DT / DT * (nstar[i] - n0) / n0;
			else if (bcon[i] == 1)
			{
				source[i] = 0.0;
			}
		}
	}
}


//================================================================================================
//====================  Incomplete Cholesky decomposition ========================================
//================================================================================================
double** INCDECOM(int TP, int* bcon, int** neigh, double** poiss, double** ic, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		if (bcon[i] != 0) continue;

		double sum = poiss[i][1];
		for (int l = 2; l <= neigh[i][1]; l++)
		{
			int j = neigh[i][l];
			if (j > i)
				continue;
			if (bcon[j] != 0)
				continue;
			sum = sum - ic[i][l] * ic[i][l];
		}
		ic[i][1] = sqrt(sum);

		for (int l = 2; l <= neigh[i][1]; l++)
		{
			int j = neigh[i][l];
			if (j < i)
				continue;
			if (bcon[j] != 0)
				continue;
			sum = poiss[i][l];
			for (int mj = 2; mj <= neigh[j][1]; mj++)
			{
				int kj = neigh[j][mj];
				if (kj >= i)
					continue;
				if (bcon[kj] != 0)
					continue;
				for (int mi = 2; mi <= neigh[i][1]; mi++)
				{
					int ki = neigh[i][mi];
					if (ki == kj)
					{
						sum = sum - ic[i][mi] * ic[j][mj];
						break;
					}
				}
			}
			ic[i][l] = sum / ic[i][1];

			for (int mj = 2; mj <= neigh[j][1]; mj++)
			{
				int kj = neigh[j][mj];
				if (i == kj)
				{
					ic[j][mj] = ic[i][l];
					break;
				}
			}
		}
	}

	return ic;
}

__global__ void incdecom(int offset, int TP, int* bcon, int* neigh, double* poiss, double* ic)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (bcon[i] == 0)
	{
		double sum = poiss[i * NEIGHBORS + 1];
		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (j > i)
				continue;
			if (bcon[j] != 0)
				continue;
			sum = sum - ic[i * NEIGHBORS + l] * ic[i * NEIGHBORS + l];
		}
		ic[i * NEIGHBORS + 1] = sqrt(sum);

		for (int l = 2; l <= neigh[i * NEIGHBORS + 1]; l++)
		{
			int j = neigh[i * NEIGHBORS + l];
			if (j < i)
				continue;
			if (bcon[j] != 0)
				continue;
			sum = poiss[i * NEIGHBORS + l];
			for (int mj = 2; mj <= neigh[j * NEIGHBORS + 1]; mj++)
			{
				int kj = neigh[j * NEIGHBORS + mj];
				if (kj >= i)
					continue;
				if (bcon[kj] != 0)
					continue;
				for (int mi = 2; mi <= neigh[i * NEIGHBORS + 1]; mi++)
				{
					int ki = neigh[i * NEIGHBORS + mi];
					if (ki == kj)
					{
						sum = sum - ic[i * NEIGHBORS + mi] * ic[j * NEIGHBORS + mj];
						break;
					}
				}
			}
			ic[i * NEIGHBORS + l] = sum / ic[i * NEIGHBORS + 1];

			for (int mj = 2; mj <= neigh[j * NEIGHBORS + 1]; mj++)
			{
				int kj = neigh[j * NEIGHBORS + mj];
				if (i == kj)
				{
					ic[j * NEIGHBORS + mj] = ic[i * NEIGHBORS + l];
					break;
				}
			}
		}
	}
}


//==================================================================================================
//========================        Boundary condition (BC)      =====================================
//==================================================================================================


void BC(int slip, int TP, int GP, int WP, int* PTYPE, int I, int** neighb, double* x, double* y, double DL, double* v, double* vstar, double* vnew, double* u, double* ustar, double* unew, double* p, double* pnew, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= GP + WP; I++)
	{
		double MINIMUM = 100;
		int k1 = TP + 1, k2 = TP + 1;
		int J;
		//-----------------------------------------------------

		if (PTYPE[I] == -1)                 // bottom
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				J = neighb[I][l];
				if ((PTYPE[J] > 0) && y[J] <= 0.0 + DL)
				{
					MINIMUM = 100;
					if (fabs(diff(I, J, x)) < MINIMUM)
					{
						k1 = J;
						MINIMUM = fabs(diff(I, J, x));
					}
				}

				if (PTYPE[J] == 0 && x[J] <= x[I] + DL / 2.0 && x[J] >= x[I] - DL / 2.0 && y[J] < 0.0 + DL)
				{
					k2 = J;
				}

			}
			v[I] = 0.0, vstar[I] = 0.0, vnew[I] = 0.0;
			u[I] = slip * u[k1], ustar[I] = slip * ustar[k1], unew[I] = slip * unew[k1];

			p[I] = p[k2], pnew[I] = pnew[k2];
		}
		//-----------------------------------------------------
		else if (PTYPE[I] == -3)                 // top of domain
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				J = neighb[I][l];
				if ((PTYPE[J] > 0) && y[J] >= 0.5 - DL)
				{
					MINIMUM = 100;
					if (fabs(diff(I, J, x)) < MINIMUM)
					{
						k1 = J;
						MINIMUM = fabs(diff(I, J, x));
					}
				}

				if (PTYPE[J] == 0 && x[J] <= x[I] + DL / 2.0 && x[J] >= x[I] - DL / 2.0 && y[J] > 0.5 - DL)
				{
					k2 = J;
				}

			}
			v[I] = 0.0, vstar[I] = 0.0, vnew[I] = 0.0;
			u[I] = slip * u[k1], ustar[I] = slip * ustar[k1], unew[I] = slip * unew[k1];
			p[I] = p[k2], pnew[I] = pnew[k2];
		}
		//-----------------------------------------------------

		else if (PTYPE[I] == -2)                 // right walls
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				J = neighb[I][l];
				if ((PTYPE[J] > 0) && x[J] >= 0.8 - DL)
				{
					MINIMUM = 100;
					if (fabs(diff(I, J, y)) < MINIMUM)
					{
						k1 = J;
						MINIMUM = fabs(diff(I, J, y));
					}
				}
				if (PTYPE[J] == 0 && y[J] <= y[I] + DL / 2 && y[J] >= y[I] - DL / 2 && x[J] >= 0.8 - DL)
				{
					k2 = J;
				}

			}

			v[I] = slip * v[k1], vstar[I] = slip * vstar[k1], vnew[I] = slip * vnew[k1];
			u[I] = 0.0, ustar[I] = 0.0, unew[I] = 0.0;
			p[I] = p[k2], pnew[I] = pnew[k2];
		}
		//-----------------------------------------------------

		else if (PTYPE[I] == -4)                 // left walls
		{
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				J = neighb[I][l];
				if ((PTYPE[J] > 0) && x[J] <= -0.4 + DL)
				{
					MINIMUM = 100;
					if (fabs(diff(I, J, y)) < MINIMUM)
					{
						k1 = J;
						MINIMUM = fabs(diff(I, J, y));
					}
				}
				if (PTYPE[J] == 0 && y[J] <= y[I] + DL / 2 && y[J] >= y[I] - DL / 2 && x[J] <= -0.4 + DL)
				{
					k2 = J;
				}

			}

			v[I] = slip * v[k1], vstar[I] = slip * vstar[k1], vnew[I] = slip * vnew[k1];
			u[I] = 0.0, ustar[I] = 0.0, unew[I] = 0.0;
			p[I] = p[k2], pnew[I] = pnew[k2];
		}
		//-----------------------------------------------------	
		else if (PTYPE[I] == 0)       // wall particles
		{
			u[I] = 0, ustar[I] = 0, unew[I] = 0;
			v[I] = 0, vstar[I] = 0, vnew[I] = 0;

		}

	}
}

__global__ void bc(int offset, int slip, int TP, int GP, int WP, int* PTYPE, int i, int* neighb, double* x, double* y, double DL, double* v, double* vstar, double* vnew, double* u,
	double* ustar, double* unew, double* p, double* pnew)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double MINIMUM = 100;
	int k1 = TP + 1, k2 = TP + 1;
	int J;
	//-----------------------------------------------------

	if (PTYPE[I] == -1)                 // bottom
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			J = neighb[I * NEIGHBORS + l];
			if ((PTYPE[J] > 0) && y[J] <= 0.0 + DL)
			{
				MINIMUM = 100;
				if (fabs(x[J] - x[I]) < MINIMUM)
				{
					k1 = J;
					MINIMUM = fabs(x[J] - x[I]);
				}
			}

			if (PTYPE[J] == 0 && x[J] <= x[I] + DL / 2.0 && x[J] >= x[I] - DL / 2.0 && y[J] < 0.0 + DL)
			{
				k2 = J;
			}

		}
		v[I] = 0.0, vstar[I] = 0.0, vnew[I] = 0.0;
		u[I] = slip * u[k1], ustar[I] = slip * ustar[k1], unew[I] = slip * unew[k1];

		p[I] = p[k2], pnew[I] = pnew[k2];
	}
	//-----------------------------------------------------
	else if (PTYPE[I] == -3)                 // top of domain
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			J = neighb[I * NEIGHBORS + l];
			if ((PTYPE[J] > 0) && y[J] >= 0.5 - DL)
			{
				MINIMUM = 100;
				if (fabs(x[J] - x[I]) < MINIMUM)
				{
					k1 = J;
					MINIMUM = fabs(x[J] - x[I]);
				}
			}

			if (PTYPE[J] == 0 && x[J] <= x[I] + DL / 2.0 && x[J] >= x[I] - DL / 2.0 && y[J] > 0.5 - DL)
			{
				k2 = J;
			}

		}
		v[I] = 0.0, vstar[I] = 0.0, vnew[I] = 0.0;
		u[I] = slip * u[k1], ustar[I] = slip * ustar[k1], unew[I] = slip * unew[k1];
		p[I] = p[k2], pnew[I] = pnew[k2];
	}
	//-----------------------------------------------------

	else if (PTYPE[I] == -2)                 // right walls
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			J = neighb[I * NEIGHBORS + l];
			if ((PTYPE[J] > 0) && x[J] >= 0.8 - DL)
			{
				MINIMUM = 100;
				if (fabs(y[J] - y[I]) < MINIMUM)
				{
					k1 = J;
					MINIMUM = fabs(y[J] - y[I]);
				}
			}
			if (PTYPE[J] == 0 && y[J] <= y[I] + DL / 2 && y[J] >= y[I] - DL / 2 && x[J] >= 0.8 - DL)
			{
				k2 = J;
			}

		}

		v[I] = slip * v[k1], vstar[I] = slip * vstar[k1], vnew[I] = slip * vnew[k1];
		u[I] = 0.0, ustar[I] = 0.0, unew[I] = 0.0;
		p[I] = p[k2], pnew[I] = pnew[k2];
	}
	//-----------------------------------------------------

	else if (PTYPE[I] == -4)                 // left walls
	{
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			J = neighb[I * NEIGHBORS + l];
			if ((PTYPE[J] > 0) && x[J] <= -0.4 + DL)
			{
				MINIMUM = 100;
				if (fabs(y[J] - y[I]) < MINIMUM)
				{
					k1 = J;
					MINIMUM = fabs(y[J] - y[I]);
				}
			}
			if (PTYPE[J] == 0 && y[J] <= y[I] + DL / 2 && y[J] >= y[I] - DL / 2 && x[J] <= -0.4 + DL)
			{
				k2 = J;
			}

		}

		v[I] = slip * v[k1], vstar[I] = slip * vstar[k1], vnew[I] = slip * vnew[k1];
		u[I] = 0.0, ustar[I] = 0.0, unew[I] = 0.0;
		p[I] = p[k2], pnew[I] = pnew[k2];
	}
	//-----------------------------------------------------	
	else if (PTYPE[I] == 0)       // wall particles
	{
		u[I] = 0, ustar[I] = 0, unew[I] = 0;
		v[I] = 0, vstar[I] = 0, vnew[I] = 0;
	}

}

//===========================================================================================
//====================  Collision of Particles computation ==================================
//===========================================================================================
void COLLISION2(int TP, double MINdistance, int* PTYPE, double Rho1, double Rho2, int** neighb, double CC, double* unew, double* vnew, double* wnew, double* x, double* y, double* z, double DT, int dim, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = 1; i <= TP; i++)
	{
		double cc;
		double ug, vg, wg, um, vm, wm, ur, vr, wr, vabs, d;
		double m1;
		double m2;

		if (PTYPE[i] == 1)m1 = Rho1; else m1 = Rho2;

		for (int l = 2; l <= neighb[i][1]; l++)
		{
			int j = neighb[i][l];
			if (PTYPE[j] == 1)m2 = Rho1; else m2 = Rho2;
			if (dim == 3)
				d = dist3d(i, j, x, y, z);
			else
				d = dist2d(i, j, x, y);

			if (i != j && d < MINdistance)
			{

				cc = CC;
				ug = (m1 * unew[i] + m2 * unew[j]) / (m1 + m2);
				vg = (m1 * vnew[i] + m2 * vnew[j]) / (m1 + m2);
				if (dim == 3)
					wg = (m1 * wnew[i] + m2 * wnew[j]) / (m1 + m2);

				ur = m1 * (unew[i] - ug);
				vr = m1 * (vnew[i] - vg);
				if (dim == 3)
					wr = m1 * (wnew[i] - wg);

				if (dim == 3)
					vabs = (ur * diff(i, j, x) + vr * diff(i, j, y) + wr * diff(i, j, z)) / d;
				else
					vabs = (ur * diff(i, j, x) + vr * diff(i, j, y)) / d;

				um = (1.0 + cc) * vabs * diff(i, j, x) / d;
				vm = (1.0 + cc) * vabs * diff(i, j, y) / d;
				if (dim == 3)
					wm = (1.0 + cc) * vabs * (z[j] - z[i]) / d;

				if (vabs > 0)
				{
					if (PTYPE[i] > 0)
					{
						unew[i] = unew[i] - um / m1;
						vnew[i] = vnew[i] - vm / m1;
						x[i] = x[i] - DT * um / m1;
						y[i] = y[i] - DT * vm / m1;
						if (dim == 3)
						{
							wnew[i] = wnew[i] - wm / m1;
							z[i] = z[i] - DT * wm / m1;
						}
					}

					if (PTYPE[j] > 0)
					{
						unew[j] = unew[j] + um / m2;
						vnew[j] = vnew[j] + vm / m2;
						x[j] = x[j] + DT * um / m2;
						y[j] = y[j] + DT * vm / m2;
						if (dim == 3)
						{
							wnew[j] = wnew[j] + wm / m2;
							z[j] = z[j] + DT * wm / m2;
						}
					}
				}
			}
		}
	}
}

__global__ void collision2(int offset, int TP, double MINdistance, int* PTYPE, double Rho1, double Rho2, int* neighb, double CC, double* unew, double* vnew, double* wnew, double* x, double* y, double* z, double DT, int dim)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double cc;
	double ug, vg, wg, um, vm, wm, ur, vr, wr, vabs, d;
	double m1;
	double m2;

	if (PTYPE[i] == 1) m1 = Rho1; else m1 = Rho2;

	for (int l = 2; l <= neighb[i * NEIGHBORS + 1]; l++)
	{
		int j = neighb[i * NEIGHBORS + l];
		if (PTYPE[j] == 1)m2 = Rho1; else m2 = Rho2;
		if (dim == 3)
			d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0) + pow((z[j] - z[i]), 2.0));
		else
			d = sqrt(pow((x[j] - x[i]), 2.0) + pow((y[j] - y[i]), 2.0));

		if (i != j && d < MINdistance)
		{
			cc = CC;
			ug = (m1 * unew[i] + m2 * unew[j]) / (m1 + m2);
			vg = (m1 * vnew[i] + m2 * vnew[j]) / (m1 + m2);
			if (dim == 3)
				wg = (m1 * wnew[i] + m2 * wnew[j]) / (m1 + m2);


			ur = m1 * (unew[i] - ug);
			vr = m1 * (vnew[i] - vg);
			if (dim == 3)
				wr = m1 * (wnew[i] - wg);

			if (dim == 3)
				vabs = (ur * (x[j] - x[i]) + vr * (y[j] - y[i]) + wr * (z[j] - z[i])) / d;
			else
				vabs = (ur * (x[j] - x[i]) + vr * (y[j] - y[i])) / d;

			um = (1.0 + cc) * vabs * (x[j] - x[i]) / d;
			vm = (1.0 + cc) * vabs * (y[j] - y[i]) / d;
			if (dim == 3)
				wm = (1.0 + cc) * vabs * (z[j] - z[i]) / d;


			if (vabs > 0)
			{
				if (PTYPE[i] > 0)
				{
					unew[i] = unew[i] - um / m1;
					vnew[i] = vnew[i] - vm / m1;
					x[i] = x[i] - DT * um / m1;
					y[i] = y[i] - DT * vm / m1;
					if (dim == 3)
					{
						wnew[i] = wnew[i] - wm / m1;
						z[i] = z[i] - DT * wm / m1;
					}
				}

				if (PTYPE[j] > 0)
				{
					unew[j] = unew[j] + um / m2;
					vnew[j] = vnew[j] + vm / m2;
					x[j] = x[j] + DT * um / m2;
					y[j] = y[j] + DT * vm / m2;
					if (dim == 3)
					{
						wnew[j] = wnew[j] + wm / m2;
						z[j] = z[j] + DT * wm / m2;
					}
				}
			}
		}
	}
}

//===========================================================================================
//====================  Calculation of DT to satisfy Courant number =========================
//===========================================================================================

double DTcalculation(double c0, double c01, double c02, double* DT, double DT_MAX, double COURANT, double DL)
{
	double max = 0;

	if (c01 > c02)c0 = c01;
	else c0 = c02;

	max = c0;
	double courant = COURANT * DL / max;
	if (DT[0] > courant) DT[0] = courant;
	else if (courant < DT_MAX)DT[0] = courant;
	else DT[0] = DT_MAX;

	return DT[0];
}

__global__ void dtCalculation(int offset, double c0, double c01, double c02, double* DT, double DT_MAX, double COURANT, double DL)
{
	double max = 0;

	if (c01 > c02)c0 = c01;
	else c0 = c02;

	max = c0;
	double courant = COURANT * DL / max;
	if (DT[0] > courant) DT[0] = courant;
	else if (courant < DT_MAX)DT[0] = courant;
	else DT[0] = DT_MAX;

}

//===========================================================================================
//================================  SPS-LES Turbulence Model ================================
//===========================================================================================

void SPS(double re, int TP, int GP, int** neighb, double* x, double* y, double* z, double coll, int KTYPE, double* unew, double* vnew, double* wnew, double n0,
	double* NEUt, double Cs, double DL, double* TURB1, double* TURB2, double* TURB3, int dim, string codeOpt)
{
	double* S12, * S11, * S22, * S13 = NULL, * S23 = NULL, * S33 = NULL;

	S12 = new double[TP + 1];
	S11 = new double[TP + 1];
	S22 = new double[TP + 1];
	if (dim == 3)
	{
		S13 = new double[TP + 1];
		S23 = new double[TP + 1];
		S33 = new double[TP + 1];
	}

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = GP + 1; i <= TP; i++)
	{
		double Uxx, Uxy, Uyx, Uyy, Uzz, Uxz, Uyz, Uzx, Uzy, S, d, w;
		double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0;
		for (int l = 2; l <= neighb[i][1]; l++)
		{
			int j = neighb[i][l];
			if (dim == 3)
				d = dist3d(i, j, x, y, z);
			else
				d = dist2d(i, j, x, y);

			if (i != j && d > coll / 10)
			{
				w = W(d, KTYPE, dim, re);
				sum1 = sum1 + (unew[j] - unew[i]) * diff(i, j, x) * w / d / d;
				sum2 = sum2 + (vnew[j] - vnew[i]) * diff(i, j, y) * w / d / d;
				sum3 = sum3 + (unew[j] - unew[i]) * diff(i, j, y) * w / d / d;
				sum4 = sum4 + (vnew[j] - vnew[i]) * diff(i, j, x) * w / d / d;
				if (dim == 3)
				{
					sum5 = sum5 + (wnew[j] - wnew[i]) * (z[j] - z[i]) * w / d / d;
					sum6 = sum6 + (unew[j] - unew[i]) * (z[j] - z[i]) * w / d / d;
					sum7 = sum7 + (vnew[j] - vnew[i]) * (z[j] - z[i]) * w / d / d;
					sum8 = sum8 + (wnew[j] - wnew[i]) * (x[j] - x[i]) * w / d / d;
					sum9 = sum9 + (wnew[j] - wnew[i]) * (y[j] - y[i]) * w / d / d;
				}
			}
		}
		Uxx = (2.0 / n0) * sum1;
		Uyy = (2.0 / n0) * sum2;
		Uxy = (2.0 / n0) * sum3;
		Uyx = (2.0 / n0) * sum4;
		if (dim == 3)
		{
			Uzz = (2.0 / n0) * sum5;
			Uxz = (2.0 / n0) * sum6;
			Uyz = (2.0 / n0) * sum7;
			Uzx = (2.0 / n0) * sum8;
			Uzy = (2.0 / n0) * sum9;
		}

		S12[i] = 0.5 * (Uxy + Uyx);
		S11[i] = 0.5 * (Uxx + Uxx);
		S22[i] = 0.5 * (Uyy + Uyy);
		if (dim == 3)
		{
			S33[i] = 0.5 * (Uzz + Uzz);
			S13[i] = 0.5 * (Uxz + Uzx);
			S23[i] = 0.5 * (Uyz + Uzy);
		}

		if (dim == 3)
			S = pow(2 * S11[i] * S11[i], 0.5) + 2 * pow(2 * S12[i] * S12[i], 0.5) + pow(2 * S22[i] * S22[i], 0.5) + pow(2 * S33[i] * S33[i], 0.5) + 2 * pow(2 * S13[i] * S13[i], 0.5) + 2 * pow(2 * S23[i] * S23[i], 0.5);
		else
			S = pow(2 * S11[i] * S11[i], 0.5) + 2 * pow(2 * S12[i] * S12[i], 0.5) + pow(2 * S22[i] * S22[i], 0.5);

		NEUt[i] = pow(Cs * DL, 2) * S;

	}

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int i = GP + 1; i <= TP; i++)
	{
		double sum1 = 0, sum2 = 0, sum3 = 0, d, w;
		for (int l = 2; l <= neighb[i][1]; l++)
		{
			int j = neighb[i][l];
			if (dim == 3)
				d = dist3d(i, j, x, y, z);
			else
				d = dist2d(i, j, x, y);

			if (i != j && d > coll / 10)
			{
				w = W(d, KTYPE, dim, re);

				sum1 = sum1 + (NEUt[j] - NEUt[i]) * diff(i, j, x) * w / d / d;
				sum2 = sum2 + (NEUt[j] - NEUt[i]) * diff(i, j, y) * w / d / d;
				if (dim == 3)
					sum3 = sum3 + (NEUt[j] - NEUt[i]) * (z[j] - z[i]) * w / d / d;
			}
		}
		if (dim == 3)
		{
			TURB1[i] = (2.0 / n0) * (2 * S11[i] * sum1 + 2 * S12[i] * sum2 + 2 * S13[i] * sum3);       
			TURB2[i] = (2.0 / n0) * (2 * S12[i] * sum1 + 2 * S22[i] * sum2 + 2 * S23[i] * sum3);
			TURB3[i] = (2.0 / n0) * (2 * S13[i] * sum1 + 2 * S23[i] * sum2 + 2 * S33[i] * sum3);
		}
		else
		{
			TURB1[i] = (2.0 / n0) * (2 * S11[i] * sum1 + 2 * S12[i] * sum2);       
			TURB2[i] = (2.0 / n0) * (2 * S12[i] * sum1 + 2 * S22[i] * sum2);
		}
	}


	delete[]S11; delete[]S12; delete[]S22;
	S11 = NULL; S12 = NULL; S22 = NULL;
	if (dim == 3)
	{
		delete[]S13; delete[]S23; delete[]S33;
		S13 = NULL; S23 = NULL; S33 = NULL;
	}
}

__global__ void turb1(int offset, double re, int* neighb, double* x, double* y, double* z, double coll, int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* NEUt, double Cs,
	double DL, double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0;
	double d, w;
	int j;
	for (int l = 2; l <= neighb[i * NEIGHBORS + 1]; l++)
	{
		j = neighb[i * NEIGHBORS + l];
		d = 0.0;
		if (dim == 3)
			d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0) + pow(z[j] - z[i], 2.0));
		else
			d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0));

		if (i != j && d > coll / 10)
		{
			w = W(d, KTYPE, dim, re);
			sum1 = sum1 + (unew[j] - unew[i]) * (x[j] - x[i]) * w / d / d;
			sum2 = sum2 + (vnew[j] - vnew[i]) * (y[j] - y[i]) * w / d / d;
			sum3 = sum3 + (unew[j] - unew[i]) * (y[j] - y[i]) * w / d / d;
			sum4 = sum4 + (vnew[j] - vnew[i]) * (x[j] - x[i]) * w / d / d;
			if (dim == 3)
			{
				sum5 = sum5 + (wnew[j] - wnew[i]) * (z[j] - z[i]) * w / d / d;
				sum6 = sum6 + (unew[j] - unew[i]) * (z[j] - z[i]) * w / d / d;
				sum7 = sum7 + (vnew[j] - vnew[i]) * (z[j] - z[i]) * w / d / d;
				sum8 = sum8 + (wnew[j] - wnew[i]) * (x[j] - x[i]) * w / d / d;
				sum9 = sum9 + (wnew[j] - wnew[i]) * (y[j] - y[i]) * w / d / d;
			}

		}
	}

	double Uxx = (2.0 / n0) * sum1;
	double Uyy = (2.0 / n0) * sum2;
	double Uxy = (2.0 / n0) * sum3;
	double Uyx = (2.0 / n0) * sum4;
	double Uzz;
	double Uxz;
	double Uyz;
	double Uzx;
	double Uzy;
	if (dim == 3)
	{
		Uzz = (2.0 / n0) * sum5;
		Uxz = (2.0 / n0) * sum6;
		Uyz = (2.0 / n0) * sum7;
		Uzx = (2.0 / n0) * sum8;
		Uzy = (2.0 / n0) * sum9;
	}

	S12[i] = 0.5 * (Uxy + Uyx);
	S11[i] = 0.5 * (Uxx + Uxx);
	S22[i] = 0.5 * (Uyy + Uyy);
	if (dim == 3)
	{
		S33[i] = 0.5 * (Uzz + Uzz);
		S13[i] = 0.5 * (Uxz + Uzx);
		S23[i] = 0.5 * (Uyz + Uzy);
	}


	double S;
	if (dim == 3)
		S = pow(2 * S11[i] * S11[i], 0.5) + 2 * pow(2 * S12[i] * S12[i], 0.5) + pow(2 * S22[i] * S22[i], 0.5) + pow(2 * S33[i] * S33[i], 0.5) + 2 * pow(2 * S13[i] * S13[i], 0.5) + 2 * pow(2 * S23[i] * S23[i], 0.5);
	else
		S = pow(2 * S11[i] * S11[i], 0.5) + 2 * pow(2 * S12[i] * S12[i], 0.5) + pow(2 * S22[i] * S22[i], 0.5);

	NEUt[i] = pow(Cs * DL, 2) * S;


}
__global__ void turb2(int offset, double re, int* neighb, double* x, double* y, double* z, double coll, int KTYPE, double n0, double* NEUt, double* TURB1, double* TURB2, double* TURB3,
	double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	double sum1 = 0, sum2 = 0, sum3 = 0;
	for (int l = 2; l <= neighb[i * NEIGHBORS + 1]; l++)
	{
		int j = neighb[i * NEIGHBORS + l];
		double d;
		if (dim == 3)
			d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0) + pow(z[j] - z[i], 2.0));
		else
			d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0));

		if (i != j && d > coll / 10)
		{
			double w = W(d, KTYPE, dim, re);

			sum1 = sum1 + (NEUt[j] - NEUt[i]) * (x[j] - x[i]) * w / d / d;
			sum2 = sum2 + (NEUt[j] - NEUt[i]) * (y[j] - y[i]) * w / d / d;
			if (dim == 3)
				sum3 = sum3 + (NEUt[j] - NEUt[i]) * (z[j] - z[i]) * w / d / d;

		}
	}
	if (dim == 3)
	{
		TURB1[i] = (2.0 / n0) * (2 * S11[i] * sum1 + 2 * S12[i] * sum2 + 2 * S13[i] * sum3);
		TURB2[i] = (2.0 / n0) * (2 * S12[i] * sum1 + 2 * S22[i] * sum2 + 2 * S23[i] * sum3);
		TURB3[i] = (2.0 / n0) * (2 * S13[i] * sum1 + 2 * S23[i] * sum2 + 2 * S33[i] * sum3);
	}
	else
	{
		TURB1[i] = (2.0 / n0) * (2 * S11[i] * sum1 + 2 * S12[i] * sum2);
		TURB2[i] = (2.0 / n0) * (2 * S12[i] * sum1 + 2 * S22[i] * sum2);
	}

}


//===========================================================================================
//=================== Avoiding particles to penetrate boundaries ============================
//===========================================================================================

void BOUNDARIE(double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL, double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, int I, double CC, int dim, string test)
{

	if (test != "turb")
	{
		if (y[I] < Ymin + DL * 3.0) { vnew[I] = fabs(vnew[I]) * CC; y[I] = Ymin + DL * 3.0; }

		if (y[I] > Ymax - DL * 3.0) { vnew[I] = -fabs(vnew[I]) * CC; y[I] = Ymax - DL * 3.0; }

		if (x[I] < Xmin + DL * 3.0) { unew[I] = fabs(unew[I]) * CC;  x[I] = Xmin + DL * 3.0; }

		if (x[I] > Xmax - DL * 3.0) { unew[I] = -fabs(unew[I]) * CC; x[I] = Xmax - DL * 3.0; }

		if (dim == 3)
		{
			if (z[I] < Zmin + DL * 3.0) { wnew[I] = fabs(wnew[I]) * CC; z[I] = Zmin + DL * 3.0; }

			if (z[I] > Zmax - DL * 3.0) { wnew[I] = -fabs(wnew[I]) * CC; z[I] = Zmax - DL * 3.0; }
		}

	}
	else
	{
		if (y[I] < Ymin + DL * 2.0) { vnew[I] = fabs(vnew[I]) * CC; y[I] = Ymin + DL * 2.0; }

		if (y[I] > Ymax - DL * 2.0) { vnew[I] = -fabs(vnew[I]) * CC; y[I] = Ymax - DL * 2.0; }



		if (x[I] > Xmax + DL * 0.0) { unew[I] = (unew[I]) * CC; x[I] = Xmin/* - DL * 3.0*/; }

		if (dim == 3)
		{
			//if (z[I]<Zmin + DL * 3.0) { wnew[I] = fabs(wnew[I])*CC; z[I] = Zmin + DL * 3.0; } 
			//if (x[I]<-0.4+DL/2.0)						            {unew[I]= fabs(unew[I])*CC;x
			//if (z[I]>Zmax - DL * 3.0) { wnew[I] = -fabs(wnew[I])*CC; z[I] = Zmax - DL * 3.0; }
		}
	}

}

void EULERINTEGRATION(int GP, int WP, int TP, double DT, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL, double Xmax, double Xmin, double Ymax, double Ymin,
	double Zmax, double Zmin, int I, double CC, int dim, string test, string codeOpt)
{

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = GP + WP + 1; I <= TP; I++)
	{
		x[I] = x[I] + unew[I] * DT;
		y[I] = y[I] + vnew[I] * DT;
		if (dim == 3)
			z[I] = z[I] + wnew[I] * DT;

		if (test != "drop")
			BOUNDARIE(x, y, z, unew, vnew, wnew, DL, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin, I, CC, dim, test);                       
	}
}

__global__ void eulerIntegration(int offset, int GP, int WP, int TP, double DT, double* x, double* y, double* z, double* unew, double* vnew, double* wnew, double DL,
	double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, int i, double CC, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	x[I] = x[I] + unew[I] * DT;
	y[I] = y[I] + vnew[I] * DT;
	if (dim == 3)
		z[I] = z[I] + wnew[I] * DT;

	if (y[I] < Ymin + DL * 3.0) { vnew[I] = fabs(vnew[I]) * CC; y[I] = Ymin + DL * 3.0; }

	if (y[I] > Ymax - DL / 2.0) { vnew[I] = -fabs(vnew[I]) * CC; y[I] = Ymax - DL / 2.0; }

	if (x[I] < Xmin + DL * 3.0) { unew[I] = fabs(unew[I]) * CC;  x[I] = Xmin + DL * 3.0; }

	if (x[I] > Xmax - DL * 3.0) { unew[I] = -fabs(unew[I]) * CC; x[I] = Xmax - DL * 3.0; }

	if (dim == 3)
	{
		if (z[I] < Zmin + DL * 3.0) { wnew[I] = fabs(wnew[I]) * CC; z[I] = Zmin + DL * 3.0; }

		if (z[I] > Zmax - DL * 3.0) { wnew[I] = -fabs(wnew[I]) * CC; z[I] = Zmax - DL * 3.0; }
	}
}


//===========================================================================================
//======================== Calculation of dynamic viscosity =================================
//===========================================================================================
double* VISCOSITY(double re, int TP, int Fluid2_type, int* PTYPE, double* MEU, double NEU1, double NEU2, double Rho1, double Rho2, int** neighb, double* x, double* y, double* z,
	int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* C, double PHI, int I, double cohes, double IIin, double yield_stressIN, double* phat, double MEU0, double N, int dim, string codeOpt)
{
	double* S12, * S11, * S22, * S13 = NULL, * S23 = NULL, * S33 = NULL/*,d, phi*/;

	//double Uxx,Uxy,Uyx,Uyy;
	//double w;
	//int i,j;

	S12 = new double[TP + 1];
	S11 = new double[TP + 1];
	S22 = new double[TP + 1];
	if (dim == 3)
	{
		S13 = new double[TP + 1];
		S23 = new double[TP + 1];
		S33 = new double[TP + 1];
	}

	//--------------  Newtonian visc ---------------------
	if (Fluid2_type == 0)
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{

			if (PTYPE[i] <= 1)MEU[i] = NEU1 * Rho1;
			if (PTYPE[i] == 2)MEU[i] = NEU2 * Rho2;

		}
	}
	//--------------------- Herschel-Bulkley Fluid  -------------------------
	if (Fluid2_type == 1)
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int i = 1; i <= TP; i++)
		{
			if (PTYPE[i] == 1)MEU[i] = NEU1 * Rho1;
			if (PTYPE[i] != 1)
			{


				double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0;
				for (int l = 2; l <= neighb[i][1]; l++)
				{
					int j = neighb[i][l];
					double d;
					if (dim == 3)
						d = dist3d(i, j, x, y, z);
					else
						d = dist2d(i, j, x, y);

					if (i != j)
					{
						double w = W(d, KTYPE, dim, re);
						sum1 = sum1 + (unew[j] - unew[i]) * diff(i, j, x) * w / d / d;
						sum2 = sum2 + (vnew[j] - vnew[i]) * diff(i, j, y) * w / d / d;
						sum3 = sum3 + (unew[j] - unew[i]) * diff(i, j, y) * w / d / d;
						sum4 = sum4 + (vnew[j] - vnew[i]) * diff(i, j, x) * w / d / d;
						if (dim == 3)
						{
							sum5 = sum5 + (wnew[j] - wnew[i]) * (z[j] - z[i]) * w / d / d;
							sum6 = sum6 + (unew[j] - unew[i]) * (z[j] - z[i]) * w / d / d;
							sum7 = sum7 + (vnew[j] - vnew[i]) * (z[j] - z[i]) * w / d / d;
							sum8 = sum8 + (wnew[j] - wnew[i]) * (x[j] - x[i]) * w / d / d;
							sum9 = sum9 + (wnew[j] - wnew[i]) * (y[j] - y[i]) * w / d / d;
						}
					}
				}
				double Uxx = (2.0 / n0) * sum1;
				double Uyy = (2.0 / n0) * sum2;
				double Uxy = (2.0 / n0) * sum3;
				double Uyx = (2.0 / n0) * sum4;
				double Uzz;
				double Uxz;
				double Uyz;
				double Uzx;
				double Uzy;
				if (dim == 3)
				{
					Uzz = (2.0 / n0) * sum5;
					Uxz = (2.0 / n0) * sum6;
					Uyz = (2.0 / n0) * sum7;
					Uzx = (2.0 / n0) * sum8;
					Uzy = (2.0 / n0) * sum9;
				}

				S12[i] = 0.5 * (Uxy + Uyx);
				S11[i] = 0.5 * (Uxx + Uxx);
				S22[i] = 0.5 * (Uyy + Uyy);
				if (dim == 3)
				{
					S33[i] = 0.5 * (Uzz + Uzz);
					S13[i] = 0.5 * (Uxz + Uzx);
					S23[i] = 0.5 * (Uyz + Uzy);
				}

				double phi = (C[I] - 0.25) * PHI / (1 - 0.25);
				if (C[I] < 0.25)phi = 0;

				double yield_stress = cohes * cos(phi) + phat[i] * sin(phi);

				double II;
				if (dim == 3)
					II = fabs((S11[i] * S22[i] * S33[i] + S12[i] * S23[i] * S13[i] + S13[i] * S12[i] * S23[i]) - (S13[i] * S22[i] * S13[i] + S11[i] * S23[i] * S23[i] + S12[i] * S12[i] * S33[i]));
				else
					II = fabs(S11[i] * S22[i] - S12[i] * S12[i]);

				MEU[i] = yield_stress / 2 / sqrt(II) + MEU0 * pow(4 * II, (N - 1) / 2);  // Chen eq.

				if (II == 0 || MEU[i] > 200) MEU[i] = 200;
			}

		}
	}
	//---------------------------------------------------------------

	delete[]S11; delete[]S12; delete[]S22;
	S11 = NULL; S12 = NULL; S22 = NULL;
	if (dim == 3)
	{
		delete[]S13; delete[]S23; delete[]S33;
		S13 = NULL; S23 = NULL; S33 = NULL;
	}

	return MEU;

}

__global__ void viscosity(int offset, double re, int TP, int Fluid2_type, int* PTYPE, double* MEU, double NEU1, double NEU2, double Rho1, double Rho2, int* neighb, double* x, double* y, double* z,
	int KTYPE, double* unew, double* vnew, double* wnew, double n0, double* C, double PHI, int I, double cohes, double II, double yield_stress, double* phat, double MEU0, double N,
	double* S11, double* S12, double* S22, double* S13, double* S23, double* S33, int dim)
{
	unsigned int i = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	//--------------  Newtonian visc ---------------------
	if (Fluid2_type == 0)
	{
		if (PTYPE[i] <= 1) MEU[i] = NEU1 * Rho1;
		if (PTYPE[i] == 2) MEU[i] = NEU2 * Rho2;
	}
	//--------------------- Herschel-Bulkley Fluid  -------------------------
	if (Fluid2_type == 1)
	{

		if (PTYPE[i] == 1) MEU[i] = NEU1 * Rho1;
		if (PTYPE[i] != 1)
		{
			double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0;
			for (int l = 2; l <= neighb[i * NEIGHBORS + 1]; l++)
			{
				int j = neighb[i * NEIGHBORS + l];
				double d;
				if (dim == 3)
					d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0) + pow(z[j] - z[i], 2.0));
				else
					d = sqrt(pow(x[j] - x[i], 2.0) + pow(y[j] - y[i], 2.0));


				if (i != j)
				{
					double w = W(d, KTYPE, dim, re);
					sum1 = sum1 + (unew[j] - unew[i]) * (x[j] - x[i]) * w / d / d;
					sum2 = sum2 + (vnew[j] - vnew[i]) * (y[j] - y[i]) * w / d / d;
					sum3 = sum3 + (unew[j] - unew[i]) * (y[j] - y[i]) * w / d / d;
					sum4 = sum4 + (vnew[j] - vnew[i]) * (x[j] - x[i]) * w / d / d;
					if (dim == 3)
					{
						sum5 = sum5 + (wnew[j] - wnew[i]) * (z[j] - z[i]) * w / d / d;
						sum6 = sum6 + (unew[j] - unew[i]) * (z[j] - z[i]) * w / d / d;
						sum7 = sum7 + (vnew[j] - vnew[i]) * (z[j] - z[i]) * w / d / d;
						sum8 = sum8 + (wnew[j] - wnew[i]) * (x[j] - x[i]) * w / d / d;
						sum9 = sum9 + (wnew[j] - wnew[i]) * (y[j] - y[i]) * w / d / d;
					}
				}
			}
			double Uxx = (2.0 / n0) * sum1;
			double Uyy = (2.0 / n0) * sum2;
			double Uxy = (2.0 / n0) * sum3;
			double Uyx = (2.0 / n0) * sum4;
			double Uzz;
			double Uxz;
			double Uyz;
			double Uzx;
			double Uzy;
			if (dim == 3)
			{
				Uzz = (2.0 / n0) * sum5;
				Uxz = (2.0 / n0) * sum6;
				Uyz = (2.0 / n0) * sum7;
				Uzx = (2.0 / n0) * sum8;
				Uzy = (2.0 / n0) * sum9;
			}

			S12[i] = 0.5 * (Uxy + Uyx);
			S11[i] = 0.5 * (Uxx + Uxx);
			S22[i] = 0.5 * (Uyy + Uyy);
			if (dim == 3)
			{
				S33[i] = 0.5 * (Uzz + Uzz);
				S13[i] = 0.5 * (Uxz + Uzx);
				S23[i] = 0.5 * (Uyz + Uzy);
			}

			double phi = (C[I] - 0.25) * PHI / (1 - 0.25);
			if (C[I] < 0.25) phi = 0;

			yield_stress = cohes * cos(phi) + phat[i] * sin(phi);
			if (dim == 3)
				II = fabs((S11[i] * S22[i] * S33[i] + S12[i] * S23[i] * S13[i] + S13[i] * S12[i] * S23[i]) - (S13[i] * S22[i] * S13[i] + S11[i] * S23[i] * S23[i] + S12[i] * S12[i] * S33[i]));
			else
				II = fabs(S11[i] * S22[i] - S12[i] * S12[i]);

			MEU[i] = yield_stress / 2 / sqrt(II) + MEU0 * pow(4 * II, (N - 1) / 2); 

			if (II == 0 || MEU[i] > 200) MEU[i] = 200;

		}
	}

}

//===========================================================================================
//================================== Prediction step ========================================
//===========================================================================================

void PREDICTION(double re, double* xstar, double* ystar, double* zstar, double* ustar, double* vstar, double* wstar, double* u, double* v, double* w, int TP, int* PTYPE, double* MEU,
	double Rho1, double Rho2, int** neighb, double* x, double* y, double* z, int KTYPE, double n0, double* phat, double* pnew, double gx, double gy, double gz, double DT, double* NEUt, double lambda,
	double* TURB1, double* TURB2, double* TURB3, double relaxp, double* RHO, double* SFX, double* SFY, int dim, string codeOpt)
{
	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		if (PTYPE[I] <= 0)
		{
			xstar[I] = x[I];
			ystar[I] = y[I];
			ustar[I] = u[I];
			vstar[I] = v[I];
			if (dim == 3)
			{
				zstar[I] = z[I];
				wstar[I] = w[I];
			}
		}
		else
		{
			double sum1 = 0;
			double sum2 = 0;
			double sum3 = 0;
			double sum4 = 0;
			double sum5 = 0;
			double sum6 = 0;
			double sum7 = 0;
			double sum8 = 0;
			double sum9 = 0;
			double NEU;

			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];
				double D;
				if (dim == 3)
					D = dist3d(I, J, x, y, z);
				else
					D = dist2d(I, J, x, y);

				if (I != J && D > 0)
				{
					if (PTYPE[I] == PTYPE[J])
					{
						if (PTYPE[I] == 1) NEU = MEU[I] / Rho1;
						else             NEU = MEU[I] / Rho2;
					}
					else
					{
						if (PTYPE[I] == 1) NEU = 2 * MEU[I] * MEU[J] / (MEU[I] + MEU[J]) / Rho1;
						else             NEU = 2 * MEU[I] * MEU[J] / (MEU[I] + MEU[J]) / Rho2;
					}

					sum1 = sum1 + (pnew[J] - phat[I]) * diff(I, J, x) * W(D, KTYPE, dim, re) / D / D;
					sum2 = sum2 + (pnew[J] - phat[I]) * diff(I, J, y) * W(D, KTYPE, dim, re) / D / D;
					if (dim == 3)
						sum3 = sum3 + (pnew[J] - phat[I]) * (z[J] - z[I]) * W(D, KTYPE, dim, re) / D / D;

					sum4 = sum4 + W(D, KTYPE, dim, re) * (u[J] - u[I]) * NEU;
					sum5 = sum5 + W(D, KTYPE, dim, re) * (v[J] - v[I]) * NEU;
					if (dim == 3)
						sum6 = sum6 + W(D, KTYPE, dim, re) * (w[J] - w[I]) * NEU;

					sum7 = sum7 + W(D, KTYPE, dim, re) * (u[J] - u[I]);
					sum8 = sum8 + W(D, KTYPE, dim, re) * (v[J] - v[I]);
					if (dim == 3)
						sum9 = sum9 + W(D, KTYPE, dim, re) * (w[J] - w[I]);

				}
			}

			ustar[I] = u[I] + gx * DT + 4 * DT * (sum4 + NEUt[I] * sum7) / (lambda * n0) + DT * TURB1[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum1;
			vstar[I] = v[I] + gy * DT + 4 * DT * (sum5 + NEUt[I] * sum8) / (lambda * n0) + DT * TURB2[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum2;
			if (dim == 3)
				wstar[I] = w[I] + gz * DT + 4 * DT * (sum6 + NEUt[I] * sum9) / (lambda * n0) + DT * TURB3[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum3;

			xstar[I] = x[I] + DT * ustar[I];
			ystar[I] = y[I] + DT * vstar[I];
			if (dim == 3)
				zstar[I] = z[I] + DT * wstar[I];

		}
	}
}


__global__ void prediction(int offset, double re, double* xstar, double* ystar, double* zstar, double* ustar, double* vstar, double* wstar, double* u, double* v, double* w, int TP, int* PTYPE,
	double* MEU, double Rho1, double Rho2, int* neighb, double* x, double* y, double* z, int KTYPE, double n0, double* phat, double* pnew, double gx, double gy, double gz, double DT, double* NEUt,
	double lambda, double* TURB1, double* TURB2, double* TURB3, double relaxp, double* RHO, double* SFX, double* SFY, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (PTYPE[I] <= 0)
	{
		xstar[I] = x[I];
		ystar[I] = y[I];
		ustar[I] = u[I];
		vstar[I] = v[I];
		if (dim == 3)
		{
			zstar[I] = z[I];
			wstar[I] = w[I];
		}
	}
	else
	{
		double sum1 = 0;
		double sum2 = 0;
		double sum3 = 0;
		double sum4 = 0;
		double sum5 = 0;
		double sum6 = 0;
		double sum7 = 0;
		double sum8 = 0;
		double sum9 = 0;

		double NEU;

		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			int J = neighb[I * NEIGHBORS + l];
			double D;
			if (dim == 3)
				D = sqrt(pow((x[J] - x[I]), 2.0) + pow((y[J] - y[I]), 2.0) + pow((z[J] - z[I]), 2.0));
			else
				D = sqrt(pow((x[J] - x[I]), 2.0) + pow((y[J] - y[I]), 2.0));


			if (I != J && D > 0)
			{

				if (PTYPE[I] == PTYPE[J])
				{
					if (PTYPE[I] == 1) NEU = MEU[I] / Rho1;
					else             NEU = MEU[I] / Rho2;
				}
				else
				{
					if (PTYPE[I] == 1) NEU = 2 * MEU[I] * MEU[J] / (MEU[I] + MEU[J]) / Rho1;
					else             NEU = 2 * MEU[I] * MEU[J] / (MEU[I] + MEU[J]) / Rho2;

				}

				sum1 = sum1 + (pnew[J] - phat[I]) * (x[J] - x[I]) * W(D, KTYPE, dim, re) / D / D;
				sum2 = sum2 + (pnew[J] - phat[I]) * (y[J] - y[I]) * W(D, KTYPE, dim, re) / D / D;
				if (dim == 3)
					sum3 = sum3 + (pnew[J] - phat[I]) * (z[J] - z[I]) * W(D, KTYPE, dim, re) / D / D;

				sum4 = sum4 + W(D, KTYPE, dim, re) * (u[J] - u[I]) * NEU;
				sum5 = sum5 + W(D, KTYPE, dim, re) * (v[J] - v[I]) * NEU;
				if (dim == 3)
					sum6 = sum6 + W(D, KTYPE, dim, re) * (w[J] - w[I]) * NEU;

				sum7 = sum7 + W(D, KTYPE, dim, re) * (u[J] - u[I]);
				sum8 = sum8 + W(D, KTYPE, dim, re) * (v[J] - v[I]);
				if (dim == 3)
					sum9 = sum9 + W(D, KTYPE, dim, re) * (w[J] - w[I]);

			}
		}

		ustar[I] = u[I] + gx * DT + 4 * DT * (sum4 + NEUt[I] * sum7) / (lambda * n0) + DT * TURB1[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum1;
		vstar[I] = v[I] + gy * DT + 4 * DT * (sum5 + NEUt[I] * sum8) / (lambda * n0) + DT * TURB2[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum2;
		if (dim == 3)
			wstar[I] = w[I] + gz * DT + 4 * DT * (sum6 + NEUt[I] * sum9) / (lambda * n0) + DT * TURB3[I] - (1 - relaxp) * (2.0 * DT / n0 / RHO[I]) * sum3;

		xstar[I] = x[I] + DT * ustar[I];
		ystar[I] = y[I] + DT * vstar[I];
		if (dim == 3)
			zstar[I] = z[I] + DT * wstar[I];

	}
}


//===========================================================================================
//========================== Calculation of pressure  ========================================
//===========================================================================================

void PRESSURECALC(int Method, int GP, int FP, int WP, int TP, int* PTYPE, double c0, double c01, double c02, double Rho1, double Rho2, double* C, double* nstar, double BETA, double n0, double* pnew, double PMIN, double PMAX,
	int IterMax, double MAXresi, double re, double* x, double* y, double* z, double coll, int KTYPE, double correction, double Rho, double relaxp, double lambda,
	double DT, double* p, double* n, int dim, int** neigh, double** poiss, int* bcon, double* source, double** ic, int imin, int imax, double eps, double* unew, double* vnew, double* wnew, bool matopt, bool srcopt, string codeOpt)
{
	//-------------- Using Equation of State instead of Poisson pressure eq. -------------------

	if (Method == 3)
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int I = GP + 1; I <= TP; I++)
		{
			if (PTYPE[I] == 2)c0 = c02;
			else c0 = c01;

			Rho = Rho2 * C[I] + Rho1 * (1 - C[I]);

			if (nstar[I] < BETA * n0)
			{
				pnew[I] = 0.0;
			}
			else
			{
				pnew[I] = (c0 * c0 * Rho / 7.0) * (pow(nstar[I] / n0, 7.0) - 1);    //P=B((rho/rho0)^7-1), B=(10*vmax)^2*Rho0/7
			}
			if (pnew[I] < PMIN)
			{
				pnew[I] = PMIN;
			}
			if (pnew[I] > PMAX)
			{
				pnew[I] = PMAX;
			}
		}
	}
	else
	{
		memset(pnew, 0, sizeof(double) * (TP + 1));
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int a = 1; a <= TP; a++)
		{
			memset(ic[a], 0, sizeof(double) * (NEIGHBORS + 1));
			memset(poiss[a], 0, sizeof(double) * (NEIGHBORS + 1));
		}
		memset(source, 0, sizeof(double) * (TP + 1));

		//---------------------- Using Conjugate gradient method -------------------

		MATRIX(re, Method, FP, WP, TP, x, y, z, coll, KTYPE, PTYPE, correction, nstar, BETA, n0, pnew, Rho1, relaxp, lambda, DT, p, n, GP, dim, neigh, poiss, bcon, source, unew, vnew, wnew, matopt, srcopt, codeOpt);

		//std::ofstream outneigh;
		//std::string filename = "outneigh_cpu1.txt";
		//outneigh.open(filename, 'w');
		//for (int j = 1; j <= TP; j++) {
		//	for (int i = 0; i < NEIGHBORS; i++) {
		//		outneigh << poiss[j][i + 1] << " ";
		//	}
		//	outneigh << std::endl;
		//}

		//outneigh.close();

		INCDECOM(TP, bcon, neigh, poiss, ic, codeOpt);

		CGM(TP, source, IterMax, MAXresi, poiss, neigh, bcon, pnew, imax, ic, DT, eps, imin, codeOpt);

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int I = 1; I <= TP; I++)
		{
			if (pnew[I] < PMIN)
			{
				pnew[I] = PMIN;
			}
			if (pnew[I] > PMAX || pnew[I] * 0.0 != 0.0)
			{
				if (pnew[I] * 0.0 != 0.0) cout << "pressao dando infinito...\n";
				pnew[I] = PMAX;
			}

		}
	}
}

__global__ void pressureCalcWC(int offset, int* PTYPE, double c0, double c01, double c02, double Rho1, double Rho2, double* C, double* nstar, double BETA, double n0, double* pnew,
	double PMIN, double PMAX, double Rho)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (PTYPE[I] == 2) c0 = c02;
	else c0 = c01;

	Rho = Rho2 * C[I] + Rho1 * (1 - C[I]);

	if (nstar[I] < BETA * n0)
	{
		pnew[I] = 0.0;
	}
	else
	{
		pnew[I] = (c0 * c0 * Rho / 7.0) * (pow(nstar[I] / n0, 7.0) - 1);    
	}

	if (pnew[I] < PMIN) pnew[I] = PMIN;

	if (pnew[I] > PMAX) pnew[I] = PMAX;

}


//===========================================================================================
//=============== Calculation of the volume of fraction if phase II in the mixture =========
//===========================================================================================
double* V_FRACTION(double re, int Fraction_method, int TP, int** neighb, int* PTYPE, double* C, int KTYPE, double* x, double* y, double* z, int dim, string codeOpt)
{
	if (Fraction_method == 1)   //Linear distribution
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int I = 1; I <= TP; I++)
		{
			double sum1 = 0;
			double sum2 = 0;
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];

				if (I != J && PTYPE[J] > 0)
				{
					sum1 = sum1 + 1;
					if (PTYPE[J] == 2)sum2 = sum2 + 1;
				}
			}
			C[I] = sum2 / sum1;
			if (sum1 == 0)C[I] = 0;


		}
	}

	if (Fraction_method == 2)   //Non linear :  Smoothed using the weight funtion
	{
		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int I = 1; I <= TP; I++)
		{
			double sum1 = 0;
			double sum2 = 0;
			for (int l = 2; l <= neighb[I][1]; l++)
			{
				int J = neighb[I][l];
				double d;
				if (dim == 3)
					d = dist3d(I, J, x, y, z);
				else
					d = dist2d(I, J, x, y);

				if (I != J && PTYPE[J] > 0)
				{
					sum1 = sum1 + W(d, KTYPE, dim, re);
					if (PTYPE[J] == 2)sum2 = sum2 + W(d, KTYPE, dim, re);
				}
			}
			C[I] = sum2 / sum1;
			if (sum1 == 0) C[I] = 0;

		}
	}
	return C;
}

__global__ void volFraction(int offset, double re, int Fraction_method, int TP, int* neighb, int* PTYPE, double* C, int KTYPE, double* x, double* y, double* z, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	if (Fraction_method == 1)
	{
		double sum1 = 0;
		double sum2 = 0;
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			int J = neighb[I * NEIGHBORS + l];

			if (I != J && PTYPE[J] > 0)
			{
				sum1 = sum1 + 1;
				if (PTYPE[J] == 2)sum2 = sum2 + 1;
			}
		}
		C[I] = sum2 / sum1;
		if (sum1 == 0)C[I] = 0;
	}

	if (Fraction_method == 2)  
	{
		double sum1 = 0;
		double sum2 = 0;
		for (int l = 2; l <= neighb[I * NEIGHBORS + 1]; l++)
		{
			int J = neighb[I * NEIGHBORS + l];
			double d;
			if (dim == 3)
				d = dist3d(I, J, x, y, z);
			else
				d = dist2d(I, J, x, y);

			if (I != J && PTYPE[J] > 0)
			{
				sum1 = sum1 + W(d, KTYPE, dim, re);
				if (PTYPE[J] == 2)sum2 = sum2 + W(d, KTYPE, dim, re);
			}
		}
		C[I] = sum2 / sum1;
		if (sum1 == 0)C[I] = 0;
	}

}


void PREPDATA(int TP, int FP, double* x, double* y, double* z, double* u, double* v, double* w, double* p, double* unew, double* vnew, double* wnew, double* pnew, double Xmin, double Ymin, double Xmax, double Ymax, double Zmin, double Zmax, int dim, string test, string codeOpt)
{

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		p[I] = pnew[I];

		if (test == "drop")
		{
			if (I >= (TP - FP))
			{
				u[I] = unew[I];
				v[I] = vnew[I];
				if (dim == 3)
					w[I] = wnew[I];

			}
		}
		else
		{
			if (I >= (TP - FP) + 1)
			{
				u[I] = unew[I];
				v[I] = vnew[I];
				if (dim == 3)
					w[I] = wnew[I];
			}
		}

		if (dim == 3)
		{
			if (p[I] * 0 != 0 || u[I] * 0 != 0 || v[I] * 0 != 0 || w[I] * 0 != 0)
			{
				cout << "ERROR#1: ERROR in particle " << I << " , x=" << x[I] << ", y=" << y[I] << ", z=" << z[I] << ", p=" << p[I] << endl;
			}

			if (x[I] >= Xmax || y[I] >= Ymax || z[I] >= Zmax || x[I] <= Xmin || y[I] <= Ymin || z[I] <= Zmin)
			{
				cout << "ERROR#2: ERROR in particle " << I << " , x=" << x[I] << ", y=" << y[I] << ", z=" << z[I] << endl;
			}
		}
		else
		{
			if (p[I] * 0 != 0 || u[I] * 0 != 0 || v[I] * 0 != 0)
			{
				cout << "ERROR#1: ERROR in particle " << I << " , x=" << x[I] << ", y=" << y[I] << ", p=" << p[I] << endl;
			}

			if (x[I] >= Xmax || y[I] >= Ymax || x[I] <= Xmin || y[I] <= Ymin)
			{
				cout << "ERROR#2: ERROR in particle " << I << " , x=" << x[I] << ", y=" << y[I] << endl;
			}
		}
	}

}

__global__ void prepData(int offset, int TP, double* x, double* y, double* z, double* u, double* v, double* w, double* p, double* unew, double* vnew, double* wnew, double* pnew, double Xmin, double Ymin, double Zmin, double Xmax, double Ymax, double Zmax, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	p[I] = pnew[I];
	u[I] = unew[I];
	v[I] = vnew[I];
	if (dim == 3)
		w[I] = wnew[I];

	if (dim == 3)
	{
		if (p[I] * 0 != 0 || u[I] * 0 != 0 || v[I] * 0 != 0 || w[I] * 0 != 0)
		{
			printf("ERROR#1: ERROR in particle %d , x=%f, y=%f, z=%f, p=%f\n", I, x[I], y[I], z[I], p[I]);
		}

		if (x[I] >= Xmax || y[I] >= Ymax || z[I] >= Zmax || x[I] <= Xmin || y[I] <= Ymin || z[I] <= Zmin)
		{
			printf("ERROR#2: ERROR in particle %d , x=%f, y=%f, z=%f\n", I, x[I], y[I], z[I]);
		}
	}
	else
	{
		if (p[I] * 0 != 0 || u[I] * 0 != 0 || v[I] * 0 != 0)
		{
			printf("ERROR#1: ERROR in particle %d , x=%f, y=%f, p=%f\n", I, x[I], y[I], p[I]);
		}

		if (x[I] >= Xmax || y[I] >= Ymax || x[I] <= Xmin || y[I] <= Ymin)
		{
			printf("ERROR#2: ERROR in particle %d , x=%f, y=%f\n", I, x[I], y[I]);
		}
	}
}


//=================================================================================================
//================================  Finding particles neighborhood ================================
//=================================================================================================
int** NEIGHBOR(double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, double re, double DELTA, int TP, double* x, double* y, double* z, int** neighb, int dim, string codeOpt)
{
	int ncx = int((Xmax - Xmin) / (re)) + 1;			 // Number of cells in x direction
	int ncy = int((Ymax - Ymin) / (re)) + 1;			 // Number of cells in y direction

	int ncz, tnc;
	if (dim == 3)
	{
		ncz = int((Zmax - Zmin) / (re + DELTA)) + 1;     // Number of cells in z direction
		tnc = ncx * ncy * ncz;							 // Total number of cells   
	}
	else
	{
		tnc = ncx * ncy;
	}

	int* Ista, * Iend;
	int* ip;

	Ista = new int[tnc + 1];
	Iend = new int[tnc + 1];
	ip = new int[TP + 1];

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int k = 1; k <= tnc; k++)
	{
		Ista[k] = 1;
		Iend[k] = 0;
	}

	for (int k = 1; k <= TP; k++)
	{
		int icell = int((x[k] - Xmin) / (re)) + 1;
		int jcell = int((y[k] - Ymin) / (re)) + 1;

		int kcell, Cnum;
		if (dim == 3)
		{
			kcell = int((z[k] - Zmin) / (re + DELTA)) + 1;
			Cnum = icell + (jcell - 1) * ncx + (kcell - 1) * ncx * ncy;
		}
		else
		{
			Cnum = icell + (jcell - 1) * ncx;       // Cell number in which particle k located
		}
		Iend[Cnum]++;						        // Number of particle in cell Cnum


		for (int m = Iend[tnc]; m >= Iend[Cnum]; m--)
		{
			if (m > 0)
			{
				ip[m + 1] = ip[m];
			}
		}

		for (int m = Cnum + 1; m <= tnc; m++)
		{
			Ista[m]++;
			Iend[m]++;
		}
		ip[Iend[Cnum]] = k;
	}

	if (codeOpt == "openmp")
	{
#pragma omp parallel for schedule (guided)
	}
	for (int I = 1; I <= TP; I++)
	{
		int icell = int((x[I] - Xmin) / (re)) + 1;
		int jcell = int((y[I] - Ymin) / (re)) + 1;
		int kcell;
		if (dim == 3)
			kcell = int((z[I] - Zmin) / (re + DELTA)) + 1;

		int k = 2;
		int row, colu, dep, m1, m2, m3, m4, m5, m6;
		if (icell == 1)m1 = 0; else m1 = -1;
		if (icell == ncx)m2 = 0; else m2 = 1;
		if (jcell == 1)m3 = 0; else m3 = -1;
		if (jcell == ncy)m4 = 0; else m4 = 1;
		if (dim == 3)
		{
			if (kcell == 1)m5 = 0; else m5 = -1;
			if (kcell == ncz)m6 = 0; else m6 = 1;

			for (row = m1; row <= m2; row++)
			{
				for (colu = m3; colu <= m4; colu++)
				{
					for (dep = m5; dep <= m6; dep++)
					{

						int Cnum = icell + row + (jcell - 1 + colu) * ncx + (kcell - 1 + dep) * ncx * ncy;

						for (int JJ = Ista[Cnum]; JJ <= Iend[Cnum]; JJ++)
						{
							int J = ip[JJ];
							if (dist3d(I, J, x, y, z) <= re + DELTA)
							{
								neighb[I][k] = J;
								k++;
							}
						}
					}
				}
			}
		}
		else
		{
			for (row = m1; row <= m2; row++)
			{
				for (colu = m3; colu <= m4; colu++)
				{
					int Cnum = icell + row + (jcell - 1 + colu) * ncx;

					for (int JJ = Ista[Cnum]; JJ <= Iend[Cnum]; JJ++)
					{
						int J = ip[JJ];
						if (I == J) continue;
						if (J < 1) continue;
						if (J > TP) continue;
						if (dist2d(I, J, x, y) < re)
						{
							neighb[I][k] = J;
							k++;
						}
					}
				}
			}
		}
		int kmax = k - 2;
		neighb[I][1] = kmax + 1;
	}

	delete[]Ista;
	delete[]Iend;
	delete[]ip;
	Ista = NULL; Iend = NULL; ip = NULL;

	return neighb;
}


__global__ void neighbor1(int offset, double Xmax, double Xmin, double Ymax, double Ymin, double re, double DELTA, int TP, double* d_x, double* d_y, int* d_neighb, int* d_Ista,
	int* d_Iend, int* d_nc, int* d_ip)
{
	unsigned int k = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	d_Ista[k] = 1;
	d_Iend[k] = 0;
}

void neighbor2gpu(int TP, double Xmin, double Ymin, double Zmin, double Xmax, double Ymax, double Zmax, double re, double DELTA, int ncx, int ncy, int tnc, double* x, double* y, double* z, int* Iend, int* Ista, int* ip, int dim, string codeOpt) {

	int icell, jcell, kcell, Cnum;

	for (int k = 1; k <= TP; k++)
	{

		icell = int((x[k] - Xmin) / re) + 1;
		jcell = int((y[k] - Ymin) / re) + 1;
		if (dim == 3)
		{
			kcell = int((z[k] - Zmin) / (re + DELTA)) + 1;
			Cnum = icell + (jcell - 1) * ncx + (kcell - 1) * ncx * ncy;
		}
		else
		{
			Cnum = icell + (jcell - 1) * ncx;       
		}
		Iend[Cnum]++;						        

		for (int m = Iend[tnc]; m >= Iend[Cnum]; m--)
		{
			if (m > 0)
			{
				ip[m + 1] = ip[m];
			}
		}

		if (codeOpt == "openmp")
		{
#pragma omp parallel for schedule (guided)
		}
		for (int m = Cnum + 1; m <= tnc; m++)
		{
			Ista[m]++;
			Iend[m]++;
		}

		ip[Iend[Cnum]] = k;
	}
}


__global__ void neighbor3(int offset, double Xmax, double Xmin, double Ymax, double Ymin, double Zmax, double Zmin, double re, double DELTA, int TP, double* d_x, double* d_y, double* d_z, int* d_neighb, int* d_Ista,
	int* d_Iend, int* d_nc, int* d_ip, int dim)
{
	unsigned int I = offset + (blockDim.x * blockIdx.x + threadIdx.x);

	int ncx = int((Xmax - Xmin) / re) + 1;     
	int ncy = int((Ymax - Ymin) / re) + 1;    
	int ncz;
	if (dim == 3)
	{
		ncz = int((Zmax - Zmin) / (re + DELTA)) + 1;
	}

	int d_icellI = int((d_x[I] - Xmin) / re) + 1;
	int d_jcellI = int((d_y[I] - Ymin) / re) + 1;
	int d_kcellI;
	if (dim == 3)
	{
		d_kcellI = int((d_z[I] - Zmin) / (re + DELTA)) + 1;
	}

	int k = 2;
	int row, colu, dep, m1, m2, m3, m4, m5, m6;
	if (d_icellI == 1)m1 = 0; else m1 = -1;
	if (d_icellI == ncx)m2 = 0; else m2 = 1;
	if (d_jcellI == 1)m3 = 0; else m3 = -1;
	if (d_jcellI == ncy)m4 = 0; else m4 = 1;
	if (dim == 3)
	{
		if (d_kcellI == 1)m5 = 0; else m5 = -1;
		if (d_kcellI == ncz)m6 = 0; else m6 = 1;

		for (row = m1; row <= m2; row++)
		{
			for (colu = m3; colu <= m4; colu++)
			{
				for (dep = m5; dep <= m6; dep++)
				{
					int Cnum = d_icellI + row + (d_jcellI - 1 + colu) * ncx + (d_kcellI - 1 + dep) * ncx * ncy;

					for (int JJ = d_Ista[Cnum]; JJ <= d_Iend[Cnum]; JJ++)
					{
						int J = d_ip[JJ];
						if (sqrt(pow(d_x[J] - d_x[I], 2.0) + pow(d_y[J] - d_y[I], 2.0) + pow(d_z[J] - d_z[I], 2.0)) <= re + DELTA)
						{
							d_neighb[(I * NEIGHBORS) + k] = J;
							k++;
						}
					}
				}
			}
		}

	}
	else
	{
		for (row = m1; row <= m2; row++)
		{
			for (colu = m3; colu <= m4; colu++)
			{
				int Cnum = d_icellI + row + (d_jcellI - 1 + colu) * ncx;

				for (int JJ = d_Ista[Cnum]; JJ <= d_Iend[Cnum]; JJ++)
				{
					int J = d_ip[JJ];
					if (I == J) continue;
					if (J < 1) continue;
					if (J > TP) continue;
					double dis = dist2d(I, J, d_x, d_y);
					if (dis < re)
					{
						d_neighb[(I * NEIGHBORS) + k] = J;
						k++;
					}
				}
			}
		}
	}

	int kmax = k - 2;
	d_neighb[(I * NEIGHBORS) + 1] = kmax + 1;
}
