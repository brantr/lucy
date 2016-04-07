#include <stdio.h>
#include <math.h>
#include "lucy.h"
#include "cuda_convolve.h"
#include "rng.h"

#define HST
#define PRIOR


Real **MakeImage(int nx, int ny);
int  *MakeRandomIndices(int nx, int nmin, int nmax);
Real **MakeFakeImage(int nx, int ny, int *ix, int *iy, Real *amp, int nr);
Real **MakeWrongImage(int nx, int ny);
Real **MakeKernel(int nx, int ny, Real sigma);
Real **ShiftImage(Real **x, int na, int nb);
Real **ModulateImage(Real **x, int nx, int ny);
Real **RebinImage(Real **x, int na, int nb, int nx, int ny);
Real **ConvolveImage(Real **image, int nx, int ny, Real **kernel, int na, int nb);

Real **LoadHST(int *nx, int *ny, int *nxp, int *nyp);

Real *MakeRandomAmplitudes(int nr, Real amin, Real amax);
int main(int argc, char **argv)
{
  int nx = 512;
  int ny = 512;
  int nxp;
  int nyp;
  int na = 128;
  int nb = 128;
  int *ix;
  int *iy;
  int nr = 1000;
  Real x;
  Real **hst;
  Real **image;
  Real **pimage;
  Real **wimage;
  Real **fimage;
  Real **fimage_hr;
  Real **fimage_lr;
  Real **fimage_a;
  Real **fimage_b;
  Real **cimage;
  Real **timage;
  Real **kernel;
  Real **kb;
  Real **kc;
  Real *amp_a;
  Real *amp_b;
  FILE *fp;
  int n_iter;

  Real sigma_a = 0.01;
  Real sigma_b = 0.001;

  Lucy L;

#ifdef HST
  n_iter = 1;
  image  = LoadHST(&nx,&ny,&nxp,&nyp);
  na = nx;
  nb = ny;
  printf("nx = %d, ny = %d\n",nx,ny);
#else
  n_iter = 20;
  image  = MakeImage(nx,ny);
  wimage = MakeWrongImage(nx,ny);
  set_rng_integer_seed(10209);
  ix = MakeRandomIndices(nr,0.1*nx,nx-0.1*nx);
  set_rng_integer_seed(7493);
  iy = MakeRandomIndices(nr,0.1*ny,ny-0.1*ny);
  set_rng_uniform_seed(12930);
  amp_a = MakeRandomAmplitudes(nr, 0.1, 10.0);
  set_rng_uniform_seed(320943);
  amp_b = MakeRandomAmplitudes(nr, 0.5, 5.0);
  fimage_a = MakeFakeImage(nx,ny,ix,iy,amp_a,nr);
  fimage_b = MakeFakeImage(nx,ny,ix,iy,amp_b,nr);
#endif  

  printf("HERE!\n");

  /* make low-res kernel */
  //kernel = MakeKernel(nx,ny,sigma_a);
  //printf("HERE!\n");
  //fflush(stdout);
  //kb     = RebinImage(kernel,nx,ny,nx,ny);
  kernel = MakeKernel(nx,ny,sigma_a);
  kb     = ShiftImage(kernel,nx,ny);
  //kernel = MakeKernel(na,nb,sigma_a);
  //kb     = RebinImage(kernel,na,nb,nx,ny);

  printf("HERE!\n");

  //deallocate_2d_array(kernel,na,nb);

  /* make high-res kernel */
  //kernel = MakeKernel(nx,ny,sigma_b);
  //kc     = RebinImage(kernel,nx,ny,nx,ny);
  //kernel = MakeKernel(na,nb,sigma_b);
  //kc     = RebinImage(kernel,na,nb,nx,ny);
  //kc = MakeKernel(nx,ny,sigma_b);

  printf("HERE (LAST)!\n");

  /* initialize the cuda convolution */
  initialize_cuda_convolve_2d(nx,ny);

  /* blur image to make "observed" */
#ifdef HST
  printf("nx = %d, ny = %d\n",nx,ny);
  cimage = cuda_convolve_2d(image, kb, nx, ny);

  
#else
  cimage = cuda_convolve_2d(fimage_a, kb, nx, ny);
  pimage = cuda_convolve_2d(fimage_b, kc, nx, ny);
  timage = cuda_convolve_2d(fimage_a, kc, nx, ny);
#endif

  //pimage = cuda_convolve_2d(image, kc, nx, ny);
  //pimage = cuda_convolve_2d(wimage, kc, nx, ny);

  /*load image an kernel*/
  L.LoadImage(  cimage, nx, ny);
  L.LoadKernel( kb,     nx, ny);

#ifdef HST
 
  pimage = ModulateImage( image, nx, ny); 
  L.LoadPriorImage( pimage, nx, ny);
#else
  L.LoadPriorImage( pimage, nx, ny);
#endif

  /*perform n_iterations of deconvolution*/
#ifndef PRIOR
  L.Deconvolve(n_iter);
#else
  L.DeconvolveWithPrior(n_iter);
#endif

#ifdef HST
  fp = fopen("hst.dat","w");  
#else
  fp = fopen("out.dat","w");  
#endif
  fwrite(&nx,1,sizeof(int),fp); 
  fwrite(&ny,1,sizeof(int),fp); 
 
  printf("image %e %e\n",image[0][0],image[0][1]);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
#ifdef HST
      fwrite(&image[i][j],1,sizeof(Real),fp); 
#else
      fwrite(&fimage_a[i][j],1,sizeof(Real),fp); 
#endif

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&L.phi_tilde[i][j],1,sizeof(Real),fp); 
      //fwrite(&cimage[i][j],1,sizeof(Real),fp); 

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&L.phi_r[i][j],1,sizeof(Real),fp); 

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&L.psi_0[i][j],1,sizeof(Real),fp); 

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&L.psi_r[i][j],1,sizeof(Real),fp); 

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&L.prior_r[i][j],1,sizeof(Real),fp); 

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
#ifdef HST
      fwrite(&image[i][j],1,sizeof(Real),fp); 
#else
      fwrite(&timage[i][j],1,sizeof(Real),fp); 
      //fwrite(&pimage[i][j],1,sizeof(Real),fp); 
#endif

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      fwrite(&kb[i][j],1,sizeof(Real),fp); 
#ifndef HST
  fwrite(&nr,1,sizeof(int),fp); 
  fwrite(&ix[0],nr,sizeof(int),fp); 
  fwrite(&iy[0],nr,sizeof(int),fp); 
  fwrite(&amp_a[0],nr,sizeof(Real),fp); 
  fwrite(&amp_b[0],nr,sizeof(Real),fp); 
#endif
  fclose(fp);

  /* clean up */ 
  destroy_cuda_convolve(); 

  return 0;
}

Real **ShiftImage(Real **x, int na, int nb)
{
  int na_1 = na/2;
  int na_2 = na - na_1;
  int nb_1 = nb/2;
  int nb_2 = nb - nb_1;
  Real rmax;
  Real **r;
  
  r = allocate_2d_array(na,nb);

  //upper right
  for(int i=0;i<na_1;i++)
    for(int j=0;j<nb_1;j++)
      r[i][j] = x[na_2+i][nb_2+j];

  //lower right
  for(int i=0;i<na_1;i++)
    for(int j=0;j<nb_2;j++)
      r[i][nb-1-j] = x[na_2+i][nb_2-j];

  //upper left
  for(int i=0;i<na_2;i++)
    for(int j=0;j<nb_1;j++)
      r[na-1-i][j] = x[na_2-i][nb_2+j];

  //lower left
  for(int i=0;i<na_2;i++)
    for(int j=0;j<nb_2;j++)
      r[na-1-i][nb-1-j] = x[na_2-i][nb_2-j];


  rmax = 0;
  Real rmin = 1.0e30;
  for(int i=0;i<na;i++)
    for(int j=0;j<nb;j++)
    {
      rmax += r[i][j];
      if(isnan(r[i][j]))
	printf("ERROR i %d j %d\n",i,j);
      if(r[i][j]<rmin)
	rmin = r[i][j];
    }
  printf("rmax = %e\n",rmax);
  printf("rmin = %e\n",rmin);

  for(int i=0;i<na;i++)
    for(int j=0;j<nb;j++)
      r[i][j]/=rmax;

  return r;
}
Real **RebinImage(Real **x, int na, int nb, int nx, int ny)
{
  int pad_x = (nx-na)/2;
  int pad_y = (ny-nb)/2;
  int na_1 = na/2;
  int na_2 = na - na_1;
  int nb_1 = nb/2;
  int nb_2 = nb - nb_1;
  Real rmax;
  Real **r;
  
  //if(nb%2)
  //nb_1++;
  //if(na%2)
  //na_1++;
  //

  printf("nx %d ny %d na_1 %d na_2 %d nb_1 %d nb_2 %d\n",nx,ny,na_1,na_2,nb_1,nb_2);
  
  r = allocate_2d_array(nx,ny);

  //upper right
  for(int i=0;i<na_1;i++)
    for(int j=0;j<nb_1;j++)
      r[i][j] = x[na_2+i][nb_2+j];

  //lower right
  for(int i=0;i<na_1;i++)
    for(int j=0;j<nb_2;j++)
      r[i][ny-1-j] = x[na_2+i][nb_2-j];

  //upper left
  for(int i=0;i<na_2;i++)
    for(int j=0;j<nb_1;j++)
      r[nx-1-i][j] = x[na_2-i][nb_2+j];

  //lower left
  for(int i=0;i<na_2;i++)
    for(int j=0;j<nb_2;j++)
      r[nx-1-i][ny-1-j] = x[na_2-i][nb_2-j];


  rmax = 0;
  Real rmin = 1.0e30;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      rmax += r[i][j];
      if(isnan(r[i][j]))
	printf("ERROR i %d j %d\n",i,j);
      if(r[i][j]<rmin)
	rmin = r[i][j];
    }
  printf("rmax = %e\n",rmax);
  printf("rmin = %e\n",rmin);

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      r[i][j]/=rmax;

  return r;
}
Real **MakeKernel(int nx, int ny, Real sigma)
{
  Real mu    = 0.5;
  Real **x = allocate_2d_array(nx,ny);
  Real xi, yj;
  Real norm = 0.0;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      xi = ((Real) i)/((Real) nx);
      yj = ((Real) j)/((Real) ny);

      x[i][j] = 1./sqrt(2.0*M_PI*sigma*sigma) * exp(-0.5*( pow(xi-mu,2) + pow(yj-mu,2) )/pow(sigma,2.));
      norm += x[i][j];
    }
  /* normalize */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      x[i][j]/=norm;

  return x;
}


Real **MakeWrongImage(int nx, int ny)
{
  Real **x = allocate_2d_array(nx,ny);
  Real ix, jy;
  Real dx, dy;

  for(int i=0;i<nx;i++)
  {
    ix = ((Real) i)/((Real) nx);

    //printf("i %d nx %d ix %e dx %e\n",i,nx,ix,dx);
    for(int j=0;j<ny;j++)
    {
      jy = ((Real) j)/((Real) ny);

      dx = fabs(ix-0.5);
      dy = fabs(jy-0.5);

      
      if( dx < 0.005)
	if( dy < 0.4)
	  x[i][j] = 1.0;

      if( dy < 0.005)
	if( dx < 0.4)
	  x[i][j] = 1.0;

      dx = fabs(ix-0.5);
      dy = fabs(jy-0.6);

    }
  }
  return x;
  
}


Real **MakeImage(int nx, int ny)
{
  Real **x = allocate_2d_array(nx,ny);
  Real ix, jy;
  Real dx, dy;

  for(int i=0;i<nx;i++)
  {
    ix = ((Real) i)/((Real) nx);

    //printf("i %d nx %d ix %e dx %e\n",i,nx,ix,dx);
    for(int j=0;j<ny;j++)
    {
      jy = ((Real) j)/((Real) ny);

      dx = fabs(ix-0.5);
      dy = fabs(jy-0.5);

      
      if( dx < 0.005)
	if( dy < 0.4)
	  x[i][j] = 1.0;

      if( dy < 0.005)
	if( dx < 0.4)
	  x[i][j] = 1.0;

      dx = fabs(ix-0.5);
      dy = fabs(jy-0.6);

      if( dy < 0.005)
	if( dx < 0.1)
	  x[i][j] = 1.0;

      dx = fabs(ix-0.5);
      dy = fabs(jy-0.4);

      if( dy < 0.005)
	if( dx < 0.1)
	  x[i][j] = 1.0;

      dx = fabs(ix-0.6);
      dy = fabs(jy-0.5);

      if( dx < 0.005)
	if( dy < 0.1)
	  x[i][j] = 1.0;

      dx = fabs(ix-0.4);
      dy = fabs(jy-0.5);

      if( dx < 0.005)
	if( dy < 0.1)
	  x[i][j] = 1.0;
    }
  }
  return x;
  
}


Real **ConvolveImage(Real **image, int nx, int ny, Real **kernel, int na, int nb)
{
  int iax;
  int jby;
  Real **cimage = allocate_2d_array(nx,ny);
  Real cmax = 0.0;
  for(int i=0;i<nx;i++)
  {
    printf("i = %d\n",i);
    fflush(stdout);
    for(int j=0;j<ny;j++)
    {
    
      for(int ia=0;ia<na;ia++)
	for(int jb=0;jb<nb;jb++)
	{
	  iax = i + ia - na/2;
	  jby = j + jb - nb/2;

	  //if(i==0&&j==0)
	   // printf("i %d j %d ia %d jb %d iax %d jby %d\n",i,j,ia,jb,iax,jby);

	  if( (iax>=0) && (jby>=0) && (iax<=nx-1) && (jby<=ny-1))
	    cimage[i][j] += kernel[ia][jb] * image[iax][jby];
	  //cimage[i][j] += kernel[ia][jb];

	
	}

      if(cimage[i][j]>cmax)
	cmax = cimage[i][j];
      
    }
  }
  printf("cmax = %e\n",cmax);

  return cimage;
}
int  *MakeRandomIndices(int nx, int nmin, int nmax)
{
  int nrand = nmax-nmin;
  int *random = (int *) malloc(nx*sizeof(int));

  for(int i=0;i<nx;i++)
    random[i] = rng_integer(nrand) + nmin;
 
  return random; 
}
Real **MakeFakeImage(int nx, int ny, int *ix, int *iy, Real *amp, int nr)
{
  Real **F = allocate_2d_array(nx,ny);
  
  for(int i=0;i<nr;i++)
    F[ix[i]][iy[i]] = amp[i];

  return F;
}
Real *MakeRandomAmplitudes(int nr, Real amin, Real amax)
{
  Real *amp = (Real *) malloc(nr*sizeof(Real));
  for(int i=0;i<nr;i++)
    amp[i] = rng_uniform(amin, amax);

  return amp;
}
Real **LoadHST(int *nx, int *ny, int *nxp, int *nyp)
{

  int xpad_a = 199;
  int ypad_a = 199;
  int xpad_b = 200;
  int ypad_b = 200;
  //int nxin = 512;
  //int nyin = 512;
  int nxin = 2001;
  int nyin = 2001;
  FILE *fp;
  char fname[200];
  Real **hst;
  Real xb;
  Real xmin = 1.0e30;
  int ib;
  int jb;
  sprintf(fname,"data/HST_cut_ascii.txt");
  *nx = nxin + xpad_a + xpad_b;
  *ny = nyin + ypad_a + ypad_b;
  *nxp = xpad_a+xpad_a;
  *nyp = ypad_a+ypad_b;

  hst = allocate_2d_array(nxin + xpad_a+xpad_b,nyin + ypad_a+ypad_b);
  fp = fopen(fname,"r");
  for(int i=0;i<nxin;i++)
    for(int j=0;j<nyin;j++)
    {
      fscanf(fp,"%d %d %f\n",&ib,&jb,&xb);
      hst[ib+xpad_b][jb+ypad_a] = xb;
      if(xb<xmin)
	xmin = xb;
      //printf("%d %d %f\n",ib,jb,hst[ib][jb]);
      if(isnan(xb))
	printf("NAN\n");
    }
  printf("Min = %e\n",xmin);

  for(int i=0;i<(*nx);i++)
    for(int j=0;j<(*ny);j++)
    {
      hst[i][j] -= xmin; 
    }
  xmin = 1.0e30;
  for(int i=0;i<(*nx);i++)
    for(int j=0;j<(*ny);j++)
    {
      if(hst[i][j]<xmin)
	xmin = hst[i][j];
      if(isnan(hst[i][j]))
      {
	printf("HST NAN! i %d j %d\n",i,j);
	fflush(stdout);	
	exit(0);
      }
    }
  printf("Min = %e\n",xmin);

/*
  for(int i=0;i<(*nx);i++)
    for(int j=0;j<(*ny);j++)
    {
      hst[i][j] = ((Real) i); 
    }
*/

  fclose(fp);

  return hst;
}
Real **ModulateImage(Real **image, int nx, int ny)
{
  Real **mimage = allocate_2d_array(nx,ny);
  Real norm_a=0;
  Real norm_b=0;
  Real norm_c=0;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      norm_a += image[i][j];
      mimage[i][j] = image[i][j]*(1.0 + 0.5*sin(4.0 * M_PI * ((Real) i)/((Real) nx))*sin(4.0 * M_PI * ((Real) j)/((Real) ny)));
      norm_b += mimage[i][j];
    }
  printf("norm_a %e norm_b %e\n",norm_a,norm_b);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      mimage[i][j] *= norm_a/norm_b;
      norm_c += mimage[i][j];
    }
  printf("norm_c %e\n",norm_c);

  return mimage;
}
