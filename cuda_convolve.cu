#include <cuda.h>
#include <cufft.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_convolve.h"
#include "global.h"

#define TPB 128

/*
cufftHandle plan;
cufftComplex *dev_x;
cufftComplex *dev_y;
cufftComplex *dev_z;
*/

int nrank;
int *n_rank;

void destroy_cuda_convolve(void)
{
  /* Destroy the CUFFT plan */
  //cufftDestroy(plan);

}
void initialize_cuda_convolve_2d(int nx, int ny)
{
  /* set rank */
  nrank = 2;

  n_rank = (int *) malloc(nrank*sizeof(int));

  n_rank[0] = nx;
  n_rank[1] = ny;
  
}

void cuda_fft_test(void)
{
  int BATCH = 1;
  int NX=512;
  int NY=512;
  int NZ=128;
  cufftHandle plan;
  cufftComplex *data ;
  //int n[3] = {NX, NY, NZ};
  int n[2] = {NX, NY};
  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*NY*NZ*BATCH);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return;
  }
  // Create a 3D FFT plan. 
  /*if ( cufftPlanMany(&plan , 3, n , NULL, 1, NX*NY*NZ, // ∗inembed, istride , idist  
					 NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
					 CUFFT_C2C , BATCH ) != CUFFT_SUCCESS ) { */
  if ( cufftPlanMany(&plan , 2, n , NULL, 1, NX*NY, // ∗inembed, istride , idist  
					 NULL, 1, NX*NY, // *onembed, ostride, odist
					 CUFFT_C2C , BATCH ) != CUFFT_SUCCESS ) { 
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return;
  }

  // Use the CUFFT plan to transform the signal in place . 
  if (cufftExecC2C(plan , data , data , CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return ;
  }

  // Inverse transform the signal in place. 
  if (cufftExecC2C(plan , data , data , CUFFT_INVERSE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
    return ;
  }
  // Note : (1) Divide by number of elements in data set to get back original data (2) Identical pointers to input and output arrays implies in−place transformation
  if (cudaThreadSynchronize() != cudaSuccess)
  {
    fprintf(stderr, "Cuda error\n");
    return;
  }

  /* Destroy the CUFFT plan . */
  cufftDestroy ( plan ); 
  cudaFree ( data ) ;
}

Real **cuda_convolve_2d(Real **x, Real **y, int nx, int ny)
{
  Real **z;

  cufftHandle plan;
  //cufftHandle iplan;
  cufftComplex *dev_x;
  cufftComplex *dev_y;
  cufftComplex *dev_z;
  cufftComplex *host_x;
  cufftComplex *host_y;
  cufftComplex *host_z;

  dim3 dimGrid(nx*ny/TPB,1,1);
  dim3 dimBlock(TPB,1,1);


  z      = allocate_2d_array(nx,ny);

  host_x = (cufftComplex *) malloc( nx*ny *sizeof(cufftComplex) );
  host_y = (cufftComplex *) malloc( nx*ny *sizeof(cufftComplex) );
  host_z = (cufftComplex *) malloc( nx*ny *sizeof(cufftComplex) );

  printf("CUDA Convolve...\n");

  cudaMalloc((void**)&dev_x, sizeof(cufftComplex)*nx*ny);
  if(cudaGetLastError() != cudaSuccess)
  {
    fprintf(stderr, "Cuda Error: Failed to allocate dev_x\n");
  }

  cudaMalloc((void**)&dev_y, sizeof(cufftComplex)*nx*ny);
  if(cudaGetLastError() != cudaSuccess)
  {
    fprintf(stderr, "Cuda Error: Failed to allocate dev_y\n");
  }

  cudaMalloc((void**)&dev_z, sizeof(cufftComplex)*nx*ny);
  if(cudaGetLastError() != cudaSuccess)
  {
    fprintf(stderr, "Cuda Error: Failed to allocate dev_z\n");
  }


  if( cufftPlanMany(&plan, nrank, n_rank, NULL, 1, nx*ny, NULL, 1, nx*ny, CUFFT_C2C, 1) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT Error: Unable to create plans\n");
    return NULL;
  }

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      host_x[nx*j + i].x = x[i][j];
      host_y[nx*j + i].x = y[i][j];

      if(isnan(x[i][j]))
      {
	printf("XNAN! i %d j %d\n",i,j);
	fflush(stdout);
	exit(0);
      }
      if(isnan(y[i][j]))
	printf("YNAN! i %d j %d\n",i,j);
    }


  cudaMemcpy(dev_x, host_x, nx*ny*sizeof(cufftComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, host_y, nx*ny*sizeof(cufftComplex), cudaMemcpyHostToDevice);

  // Transform in place 
  if( cufftExecC2C(plan, dev_x, dev_x, CUFFT_FORWARD) != CUFFT_SUCCESS )
  {
    fprintf(stderr, "CUFFT Error: ExecC2C Forward on dev_x failed %d\n",cudaGetLastError());
    return NULL;
  }

  // Transform in place 
  if( cufftExecC2C(plan, dev_y, dev_y, CUFFT_FORWARD) != CUFFT_SUCCESS )
  {
    fprintf(stderr, "CUFFT Error: ExecC2C Forward on dev_y failed %d\n",cudaGetLastError());
    return NULL;
  }

  printf("Before convolution nx = %d ny = %d\n",nx,ny);
  // do convolution 
  convolve_2d_CUDA<<<dimGrid,dimBlock>>>(dev_x, dev_y, dev_z, nx, ny);

  // Transform in place 

  if( cufftExecC2C(plan, dev_z, dev_z, CUFFT_INVERSE) != CUFFT_SUCCESS )
  {
    fprintf(stderr, "CUFFT Error: ExecC2C Inverse on dev_z failed %d\n",cudaGetLastError());
    return NULL;
  }


  cudaThreadSynchronize();

  // To CPU 
  cudaMemcpy(host_z, dev_z, nx*ny*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      if(isnan(x[i][j]))
      {
	printf("XNAN! i %d j %d\n",i,j);
	fflush(stdout);
	exit(0);
      }
      if(x[i][j]<0)
      {
	printf("XNEG! i %d j %d\n",i,j);
	fflush(stdout);
	exit(0);
      }
    }
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      if(y[i][j]<0)
      {
	printf("YNEG! i %d j %d\n",i,j);
	fflush(stdout);
	exit(0);
      }
      if(isnan(y[i][j]))
      {
	printf("YNAN! i %d j %d\n",i,j);
	fflush(stdout);
	exit(0);
      }
    }
    
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      z[i][j] = host_z[nx*j + i].x;
      if(isnan(z[i][j]))
      {
	printf("ZNAN! i %d j %d\n",i,j);
	fflush(stdout);
//	exit(0);
      }
    }

  cufftDestroy(plan);

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);

  free(host_x);
  free(host_y);
  free(host_z);

  // To GPU 
  // return result 
  return z;
  
}

__global__ void convolve_2d_CUDA(cufftComplex *dev_x, cufftComplex *dev_y, cufftComplex *dev_z, int nx, int ny)
{
  Real scale = (Real) (1./( ((Real) nx)*((Real) ny) ));
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx<nx*ny)
  {
    dev_z[idx].x = (dev_x[idx].x*dev_y[idx].x - dev_x[idx].y*dev_y[idx].y)*scale;
    dev_z[idx].y = (dev_x[idx].x*dev_y[idx].y + dev_x[idx].y*dev_y[idx].x)*scale;
    //dev_z[idx].x = dev_x[idx].x;
    //dev_z[idx].y = dev_x[idx].y;
    //dev_z[idx].x = 1;
    //dev_z[idx].y = 0;
  }
}
