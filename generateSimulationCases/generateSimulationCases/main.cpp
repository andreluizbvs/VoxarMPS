#include "genParticlesFunctions.h"
#include <omp.h>

void main(){

	bool compressible = true;										//         Running weakly compressible = true or fully incompressible = false simulations
	double DL;														//         Average particle distance
	double multiplier = 48;												//		   Multiplier to change simulation size (increase number of particles)
	if (!compressible) DL = 0.010;									
	else DL = 0.0040;
	const char filename[] = "input-dam.grd";
	//createParticlesOilSpill2D(DL, multiplier, filename);
	createParticlesDamBreak2D(DL, multiplier, filename);
	
	std::cout << "Done!\n";
}
