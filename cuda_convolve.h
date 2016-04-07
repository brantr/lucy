#ifndef CUDA_CONVOLVE_H
#define CUDA_CONVOLVE_H
#include<cuda.h>
#include<cufft.h>
#include"global.h"
/*
extern cufftHandle plan;
extern cufftComplex *dev_x;
extern cufftComplex *dev_y;
extern cufftComplex *dev_z;
*/
extern int nrank;
extern int *n_rank;

void cuda_fft_test(void);

void destroy_cuda_convolve(void);
void initialize_cuda_convolve_2d(int nx, int ny);
Real **cuda_convolve_2d(Real **x, Real **y, int nx, int ny);

__global__ void convolve_2d_CUDA(cufftComplex *dev_x, cufftComplex *dev_y, cufftComplex *dev_z, int nx, int ny);

#endif /* CUDA_CONVOLVE_H */
