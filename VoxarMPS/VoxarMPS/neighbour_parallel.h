#ifndef NEIGHBOUR_PARALLEL_H    
#define NEIGHBOUR_PARALLEL_H

//#include "stdafx.h"
//#include <SFML/Graphics.hpp>
//VS Headers


void neighbour_cuda_2d(int TP, double *x, double *y, double DELTA, double re, int ** neighb, double Xmin, double Xmax, double Ymin, double Ymax);
void neighbour_cuda_2d_gpu(int TP, double *d_x, double *d_y, double DELTA, double re, int *d_neighb, double Xmin, double Xmax, double Ymin, double Ymax);


#endif