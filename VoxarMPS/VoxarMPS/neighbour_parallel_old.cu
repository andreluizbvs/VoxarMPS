#include <algorithm>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <time.h>
#include <vector>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "neighbour_parallel.h"
#include "common.h"
#include "device_launch_parameters.h"





//VS headers
//still need to make x,y,z compatible with main
//add tests

//in mps the x array starts from 1 index
//you have to caliberate everything according to that and test according to that. 
//after that you run the code and time it
//after that you run a dam break case 
//after that you classify the code 


// 2d code

// ----------------- CUDA KERNELS -------------------------

__global__ void calcHash(double *d_x, double *d_y,  int *d_particleHash,\
	int *d_TP, double *d_Xmax, double *d_Xmin, double *d_re, double *d_DELTA, double *d_Ymin, \
  double *d_Ymax, int *d_particleid, int *d_tnc, int *ncx, \
  int *ncy){

  int k = threadIdx.x + blockIdx.x * blockDim.x;
  if(k < *d_TP){


  *ncx = int((*d_Xmax - *d_Xmin) / (*d_re + *d_DELTA)) + 1;     // Number of cells in x direction
  *ncy = int((*d_Ymax - *d_Ymin) / (*d_re + *d_DELTA)) + 1;     // Number of cells in y direction
  *d_tnc = *ncx * *ncy ;

  
  int *icell, *jcell, *cellNum;

  int sizeint = sizeof(int);
  icell = (int *)malloc(sizeint);
  jcell = (int *)malloc(sizeint);
  cellNum = (int *)malloc(sizeint);
  
  *icell = int((d_x[k+1] - *d_Xmin) / (*d_re + *d_DELTA)) + 1;
  *jcell = int((d_y[k+1] - *d_Ymin) / (*d_re + *d_DELTA)) + 1;

  *cellNum = *icell + (*jcell - 1)* *ncx;

  d_particleHash[k] = *cellNum;
  d_particleid[k] = k+1;
  //particlehash and particleId have indices starting from 0 which corresponds to the index 1 in the coordinate array (which is x, y, and z)
  //bsically x and y will have sizes TP+1
  free(icell);
  free(jcell);
  free(cellNum);
}

}

__global__ void findCellStart(int *particleHash, int *cellStart, int *cellEnd, int *TP){

  int k = threadIdx.x + blockIdx.x * blockDim.x; // here index value corresponds to index of the array particleHash
  //cellNum will assume values starting from 1 but the corresponding cellstart and cellend will start from first index as 0
  if(k < *TP){
  if (particleHash[k] != particleHash[k+1] && k!= *TP - 1){
    cellEnd[particleHash[k] - 1] = k;
    cellStart[particleHash[k+1] - 1] = k+1;
  }
  if(k == *TP - 1){
    cellEnd[particleHash[k] - 1] = k;
  }
    }

  free(&k);            
}

__global__ void createNeighbourArraysCUDA(int *d_neighb, int *cellStart, int *cellEnd, int *particleHash, int *particleid, int *ncx, int *ncy, int *d_max_neighb,  int *d_TP, double *d_re, double *d_DELTA, double *d_x, double *d_y){


  int index = threadIdx.x + blockIdx.x * blockDim.x; 

  if(index < *d_TP){
  int pid, icell, jcell, cellNum,  neighb_index, Cnum, J,  curr_neighb_num,  m1, m2, m3, m4;
  
  double R;
 
  cellNum = particleHash[index]; 
  pid = particleid[index];
  
  neighb_index = (pid-1) * (*d_max_neighb + 1);

  jcell = ((cellNum - 1)/ *ncx) + 1;
  icell = cellNum - *ncx * (jcell - 1);
 
  curr_neighb_num = 0;
  
  if (icell == 1) m1 = 0; else m1 = -1;
  if (icell == *ncx) m2 = 0; else m2 = +1;
  if (jcell == 1) m3 = 0; else m3 = -1;
  if (jcell == *ncy) m4 = 0; else m4 = +1;

  for (int row = m1; row <= m2; row++)
  {
    for (int colu = m3; colu <= m4; colu++) 
    {
      

        Cnum = icell + row + (jcell - 1 + colu)* *ncx;

        if (cellEnd[Cnum - 1] != -1){

        for (int JJ = cellStart[Cnum -1]; JJ <= cellEnd[Cnum - 1]; JJ++)
        {
          J = particleid[JJ];
          R = sqrt(pow(d_x[J]-d_x[pid], 2.0) + pow(d_y[J]-d_y[pid], 2.0));
          if (R <= *d_re + *d_DELTA){
          curr_neighb_num = curr_neighb_num + 1;
          d_neighb[neighb_index + curr_neighb_num] = J; //here the index is shifted by one unit to conform to the original MPS convention
          }
        }
      }
      
    }
  }
  
  
  d_neighb[neighb_index] = curr_neighb_num;

   }
 }



__global__ void InitializeCellDetails(int *cellStart, int *cellEnd, int *d_tnc){
  int index = threadIdx.x + blockIdx.x * blockDim.x; 
  if(index < *d_tnc){
  cellStart[index] = 0; cellEnd[index] = -1;
}
free(&index);
}




// ------------------------- Host sub-sub-routine for neighbour computation ------------------------ 




void neighbour_cuda_2d(){

  int MAX_NEIGHB = 150, THREADS_PER_BLOCK = 512;

  //cout<<endl<<"Time study for neighbour_cuda_1()"<<endl;

  // ------------------ variable declarations and initializations ------------------------------

  int *d_cellEnd, *d_cellStart, *d_TP, *d_tnc, *tnc, *d_ncx, *d_ncy, *d_max_neighb;
  int *d_particleHash, *d_particleid, *d_neighb, *h_neighb, *d_sizeof_neighbours;
  double *d_x, *d_y, *d_Xmax, *d_Xmin, *d_Ymax, *d_Ymin, *d_re, *d_DELTA;

  int arrsizeint = TP * sizeof(int);
  int sizeint = sizeof(int);
  int arrsizedouble = (TP+1) * sizeof(double);
  int sizedouble = sizeof(double);
  int sizeneighb = TP * (MAX_NEIGHB + 1) * sizeof(int);
  int sizeof_neighbours = (MAX_NEIGHB + 1) * sizeof(int);

  tnc = (int *)malloc(sizeint);
  h_neighb = (int *)malloc(sizeneighb);

  

  cudaMalloc((void **)&d_particleHash, arrsizeint);
  cudaMalloc((void **)&d_particleid, arrsizeint); 
  cudaMalloc((void **)&d_x, arrsizedouble);
  cudaMalloc((void **)&d_y, arrsizedouble);
  cudaMalloc((void **)&d_Xmin, sizedouble);
  cudaMalloc((void **)&d_Xmax, sizedouble);
  cudaMalloc((void **)&d_Ymin, sizedouble);
  cudaMalloc((void **)&d_Ymax, sizedouble);
  cudaMalloc((void **)&d_re, sizedouble);
  cudaMalloc((void **)&d_DELTA, sizedouble);
  cudaMalloc((void **)&d_TP, sizeint);
  cudaMalloc((void **)&d_tnc, sizeint);
  cudaMalloc((void **)&d_ncx, sizeint);
  cudaMalloc((void **)&d_ncy, sizeint);
  cudaMalloc((void **)&d_neighb, sizeneighb);
  cudaMalloc((void **)&d_max_neighb, sizeint);
  cudaMalloc((void **)&d_sizeof_neighbours, sizeof_neighbours);

  cudaMemcpy(d_x, x, arrsizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, arrsizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Xmin, &Xmin, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Xmax, &Xmax, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ymin, &Ymin, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ymax, &Ymax, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_re, &re, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_DELTA, &DELTA, sizedouble, cudaMemcpyHostToDevice);
  cudaMemcpy(d_TP, &TP, sizeint, cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_neighb, &MAX_NEIGHB, sizeint, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizeof_neighbours, &sizeof_neighbours, sizeint, cudaMemcpyHostToDevice);
 

  

  // --------------- running the calcHash kernel ----------------------------------------

  calcHash<<<TP/THREADS_PER_BLOCK + 1,THREADS_PER_BLOCK>>>(d_x, d_y, d_particleHash, d_TP, d_Xmax, d_Xmin, d_re, d_DELTA, d_Ymin, d_Ymax, d_particleid, d_tnc, d_ncx, d_ncy);
  
  // ---------------- sorting the particleHash array -----------------------------

  thrust::device_ptr<int> dev_Hash(d_particleHash);
  thrust::device_ptr<int> dev_id(d_particleid);
  thrust::sort_by_key(dev_Hash, dev_Hash + TP, dev_id); //need to generalise this 10

 
  
  // --------------------- finding cell start and cell end for each cell -----------------------------

  cudaMemcpy(tnc, d_tnc, sizeint, cudaMemcpyDeviceToHost);
  int cellarrsize = *tnc * sizeof(int);
 
  cudaMalloc((void **)&d_cellStart, cellarrsize); 
  cudaMalloc((void **)&d_cellEnd, cellarrsize); 


  InitializeCellDetails<<<*tnc/THREADS_PER_BLOCK + 1,THREADS_PER_BLOCK>>>(d_cellStart, d_cellEnd, d_tnc);
  findCellStart<<<TP/THREADS_PER_BLOCK + 1,THREADS_PER_BLOCK>>>(d_particleHash, d_cellStart, d_cellEnd, d_TP);
 
  
  // -------------------------- Creating neighbour arrays for each particle ------------------------------

 
  createNeighbourArraysCUDA<<<TP/THREADS_PER_BLOCK + 1,THREADS_PER_BLOCK>>>(d_neighb, d_cellStart, d_cellEnd, d_particleHash, d_particleid, d_ncx, d_ncy, d_max_neighb, d_TP, d_re, d_DELTA, d_x, d_y);
 
  
  
  cudaMemcpy(h_neighb, d_neighb, sizeneighb, cudaMemcpyDeviceToHost);



  // ---------------------------- Populating neighb array ----------------------
  
  
  for(int j=0; j<TP; j++){
    for(int i=0; i<h_neighb[j*(MAX_NEIGHB + 1)]; i++){
      neighb[j+1][i+2] = h_neighb[j*(MAX_NEIGHB + 1) + i + 1];
    }
    neighb[j+1][1] = h_neighb[j*(MAX_NEIGHB + 1)]+1;
  }
  

  
  
  // -------------------------- Deallocating memory ---------------------------

  cudaFree(d_particleHash);
  cudaFree(d_particleid);
  cudaFree(d_cellStart);
  cudaFree(d_cellEnd);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_Xmin);
  cudaFree(d_Xmax);
  cudaFree(d_Ymin);
  cudaFree(d_Ymax);
  cudaFree(d_re);
  cudaFree(d_TP);
  cudaFree(d_tnc);
  cudaFree(d_ncx);
  cudaFree(d_ncy);
  cudaFree(d_neighb);
  cudaFree(d_max_neighb);
  cudaFree(d_sizeof_neighbours);

  free(h_neighb);
  free(tnc);
}