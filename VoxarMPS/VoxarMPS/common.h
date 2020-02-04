#ifndef COMMON_H
#define COMMON_H


//_______________________ Global Variables Definition ____________________________________

extern int I, J, K, NUM, l;

extern double n0, Rho, MEUi, MEUj, NEU;

extern double *x, *y, *p, *u, *v;
extern int *PTYPE;
extern double *xstar, *ystar, *xnew, *ynew, *ustar, *vstar, *pnew, *phat, *unew, *vnew, *NEUt, *TURB1, *TURB2, *n, *nstar, *nnew, *MEU, *C, *RHO, *RHOnew, *RHOSMOOTH, *MEU_Y, *p_rheo, *p_rheo_new, *II, *Inertia;
extern double *c, *cmin;
extern int *F_S;
extern double *FSD, *nvx, *nvy;
extern double *Lambda_X, *Lambda_Y, *DivV, *DivVnew, *DivVstar;
extern double *L_11, *L_12, *L_21, *L_22;
extern double *axc, *ayc, *axp, *ayp, *pstar;
extern double *Tau_xx, *Tau_yy, *Tau_xy;
extern int    **neighb;
extern int    p_count;
extern double lambda;                          //         MPS discretization coefficient
extern double Xmin;						    //         Minimum x of searching grid
extern double Ymin;						    //         Minimum y of searching grid
extern double Xmax;						    //         Maximum x of searching grid
extern double Ymax;						    //         Maximum y of searching grid
extern int    FP;							    //         Number of Fluid particles: in this model FP calculate in each time step
extern int    WP;							    //         Number of wall particles
extern int    GP;								//         Number of ghost particles
extern double maxFA;
extern double maxVIS;
extern double DT;                  //         Initial Time step size
extern double DTSTAR;
extern double Timestep;
extern double c0;
extern double  ShiftDX;						//			Shifting method in X direction
extern double  ShiftDY;						//			Shifting method in Y direction
extern double *PSR;							//			Particles Shifting ratio


//****************************************************************
//*************  Assigning Model Parameters **********************
//****************************************************************


//*************** basic model conditions  ************************
extern double DL;						  //         Average particle distance (or particle size)
extern double re;                        //		 Support area radius
extern double BETA;                       //         Coefficient for determination of free surface
extern double relaxp;					  //         Relaxation factor for pressure correction
extern double COURANT;                    //         CFL number
extern double Cs;                         //         Smogorinsky Constant (For using in SPS-LES turbulence model)
extern double DELTA;                  //         Expansion value of background grid cells size.
extern int    TP;						  //         Total number of particles
extern int    KTYPE;                         //         Kernel type
extern int    DIM;                           //         Dimension
extern int    TURB;                          //         TURB=0: NO turbulence model, TURB=1: SPS turbulence model
extern int    Fraction_method;               //         Method of calculation of volume of fraction. 1: Linear dist across the extern interface, 2: smoothed value

extern int	   TS_Method;					  //	     Time Stepping Method. (Shakibaeinia splitting (P_C) method =1, Monaghan predictor-corrector method=2, Symplectic method =3)
extern double Ncorrection;                 //         Correction of n0 value to improve the incompressibility and initial pressure fluctuation

//*************** basic flow conditions  ***************************
extern double NEU1;					 //         Kinematic Viscosity
extern double NEU2;					 //         Kinematic Viscosity
extern double Rho1;						 //         Density of phase 1
extern double Rho2;						 //         Density of phase 2
extern double gx;                         //         Gravity acceleration in x direction
extern double gy;                       //         Gravity acceleration in y direction
extern double VMAX;	 					 //         To avoid jumping particles out of domain (If v>Vmax ---> v=0)

//*************** pressure and pressure gradient Calculation parameters *****************
extern int    Method;						//         Fully incompressible MPS: Method=1; Fully incompressible M-MPS: Method=2; Weakly compressible: Method=3 .
extern double c01;                        //         Numerical sound speed fluid 1. (Weakly compressible model)
extern double c02;                        //         Numerical sound speed fluid 2. (Weakly compressible model)

extern int    KHcorrection;                //         Khayyer and Gotoh pressure correction 2009 (1=yes, 0=no)
extern int    MPGRAD;						//		   Modified Pressure Gradient (Two methods)
extern int	   OGERcorrection;				//		   If =1 then Oger correction is applied in the pressure gradient calculations

extern int	   Cs_DYN;					   //		   If = 0 then the Cs is constant in the pressure calculation over time.
extern double PMAX;                    //          A limit for the value of calculated pressure to avoid program crashing
extern double PMIN;                      //          Minimum pressure, to avoid a high negative pressure
extern int    SmoothWP;				   //		   If = 1 then the pressure field for the Wall particles is smoothed by the kernel approximation
extern double GAMA;

//******************* Boundary Correction  **************************
extern int    WBC;                             //         Type of wall B.C.  No-Slip:0 & -1, Slip:1
extern double COL;                     //        Minimum particle distance to prevent collision at boundaries
extern double CE;                            //        Collision coefficient for partilces near boundaries

//***************************************************************************************
extern int DT_DYN;
extern int	ART_VISCO;
extern int MPS_HL;						    //		 Higher order Laplacian term of the velocity field / PNUM (if =1 then yes)
extern int	DivVel;							//		 Role of compressibility (Divergence of the Velocity !=0) in the shear stress term of the momentum equation (stress_cal_method = 1)

//*************** Particle number density calculation ***********************************
extern int PNum_HT;						//		Higher order time diff. of the kernel function (Khayyer 2009, 2011)
extern int Continuity;					    //		If =1 , continuity equation is solved for NEW PNUM. If =0, NEW PNUM is directly calculated from kernel function (Wij)
extern double DiffPNUM;					//		The Laplacian of the PNUM is added to the continuity equation. If >=0 then it is activated

//*************** Collision Method parameters   *********************
extern int Collision;						  //		Activating the Collision Method (If =1 then it is activated)
extern double coll;                   //        Minimum particle distance to prevent collision
extern double CC;                            //        Collision coefficient related to the kenetic energy of the collision

//*********************** Repulsive Force  ******************************
extern int Repulsive;						  //		Activating the Repulsive Force-Artificial Pressure (If =1 then it is activated)
extern double epsn;
extern double epsp;
extern int n_repul;

//*************** Shifting Algorithm parameters   *********************
extern int Shift;						     //		Activating the Shifting Algorithm (If =1 then it is activated)
extern int UpdateVEL;
extern double A;							 //     Shifting Algorithm Coefficient, Default=2
extern double FSC;						     //		Free Surface Correction implementation in the shifting algorithm (Yes If = 1)

//*************** Freesurface Detection   *********************
extern double FSD_LL;					 //		Lower limit of n[i]/n0 ratio for free surface detection (FSD)
extern double FSD_HL;					 //		upper limit of n[i]/n0 ratio for free surface detection (FSD)
extern int PNUMgrad;						 //		Determining the method that is used for the PNUM gradient calculation (0,1,2)

//****************** Rheological parameters   *************************-
extern int    Fluid2_type;                   // Newtonian:0  , H-B fluid:1
extern double N;							  // flow behaviour (power law) index
extern double MEU0;						  // consistency index
extern double PHI;                       // friction angle (RAD)
extern double PHI_wall;                   // friction angle (RAD)
extern double PHI_bed;                    // friction angle (RAD)
extern double PHI_2;                       // second friction angle Values are based on  Minatti &  Paris (2015)
extern double cohes;						  // cohesiveness coefficient
extern double visc_max;                     // maximum viscosity uses to avoid singularity
extern double dg;                       // grain size
extern double I0;                          // I0 value in Meu9I0 rheology     Values are based on  Minatti &  Paris (2015)
extern double mm;
extern int    stress_cal_method;             // Method 1; viscosity is directly used in momentum equation. Method 2: first the stress tensor is calculated then it used in momentum equation
extern double yeild_stress;

//*************** Time parameters  ************************************-
extern double  t, T;                           //         Simulation time (sec)
extern double  DT_MAX;                    //         Maximum size of time extern interval allowed
extern double  tec_out_intrval;            // time extern interval for tecplot output

#endif