#include"lucy.h"
#include"cuda_convolve.h"
Lucy::Lucy(void)
{
}
void Lucy::SaveInitialImage(void)
{
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      psi_0[i][j] = psi_r[i][j];
}

void Lucy::DeconvolveWithPrior(int n_iter)
{
  Real norm      = 0.0;
  Real psi_r_tot = 0.0;
  /* normalize psi_r */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      norm  += phi_tilde[i][j];
      psi_r[i][j] = prior_r[i][j];
      psi_r_tot  += prior_r[i][j];
    }
  printf("norm = %e\n",norm);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      psi_r[i][j]   *= norm/psi_r_tot;
      prior_r[i][j] *= norm/psi_r_tot;
    }
  norm = 0.0;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      norm += psi_r[i][j];
    }
  printf("norm (after) = %e\n",norm);


  SaveInitialImage();

  r = 0;
  while(r<n_iter)
  {
    UpdateEstimate();
    printf("r %d\n",r);
  }

  return;
}

void Lucy::Deconvolve(int n_iter)
{
  SaveInitialImage();
  r = 0;
  while(r<n_iter)
  {
    printf("r %d\n",r);
    UpdateEstimate();
  }

  return;
}
void Lucy::LoadImage(Real **phi_in, int nx_in, int ny_in)
{
  Real norm = 0;
  Real psi_r_norm = 0;
  /* allocate arrays */
  nx    = nx_in;
  ny    = ny_in;
  phi_r	    = allocate_2d_array(nx,ny);
  phi_tilde = allocate_2d_array(nx,ny);
  Q	    = allocate_2d_array(nx,ny);
  psi_r	    = allocate_2d_array(nx,ny);
  psi_0	    = allocate_2d_array(nx,ny);

  

  /* load image */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {
      phi_tilde[i][j] = phi_in[i][j];

      norm += phi_in[i][j];

      /* make initial guess */
      psi_r[i][j] = 1.;

      psi_r_norm += psi_r[i][j];
    }

  /* normalize psi_r */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      psi_r[i][j] *= norm/psi_r_norm;

  /* set iteration to zero */
  r = 0;
}
void Lucy::LoadPriorImage(Real **prior_in, int nx_in, int ny_in)
{
  prior_r = allocate_2d_array(nx,ny);

  /* load image */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      prior_r[i][j] = prior_in[i][j];
}

void Lucy::LoadKernel(Real **A_in, int na_in, int nb_in)
{
  
  /* allocate kernel */
  na = na_in;
  nb = nb_in;
  A  = allocate_2d_array(na,nb);

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      A[i][j] = A_in[i][j];
}

void Lucy::UpdateEstimate(void)
{

  /* first, calculate phi_r*/
  calculate_phi_r();

  /* then, find Q_r */
  calculate_Q_r();

  /* then, update psi_r */
  update_psi_r();

  /* iterate r */
  r++;

}

void Lucy::calculate_phi_r(void)
{
  int ia_min;
  int ia_max;
  int jb_min;
  int jb_max;
  int iax;
  int jby;

  //Real **pconv = cuda_convolve_2d(phi_tilde,A,nx,ny);
  Real **pconv = cuda_convolve_2d(psi_r,A,nx,ny);

  Real pmax = 0;
  /* loop over x and y */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {

      /* reset phi_r */
      phi_r[i][j] = pconv[i][j];

    }
  printf("pmax = %e\n",pmax);
  deallocate_2d_array(pconv,nx,ny);

  return;

      /* find limits of convolution integral */

      /*ia_min = find_ia_min(i,j);
      ia_max = find_ia_max(i,j);
      jb_min = find_jb_min(i,j);
      jb_max = find_jb_max(i,j);*/

      /* loop over kernel */
      /*for(int ia=ia_min;ia<ia_max;ia++)
	for(int jb=jb_min;jb<jb_max;jb++)*/
/*
      for(int ia=0;ia<na;ia++)
	for(int jb=0;jb<nb;jb++)
	{
	  iax = i + ia - na/2;
	  jby = j + jb - nb/2;

	  if( (iax>=0) && (jby>=0) && (iax<=nx-1) && (jby<=ny-1))
	    phi_r[i][j] += psi_r[iax][jby] * Q[ia][jb];
	}
*/
}

void Lucy::calculate_Q_r(void)
{
  int i_min;
  int i_max;
  int j_min;
  int j_max;
  int iax;
  int jby;

  Real Qmax = 0;

  //eqn 13 of lucy 1974
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      if(phi_r[i][j]>0)
      {
	Q[i][j] = phi_tilde[i][j] / phi_r[i][j];
	if(Q[i][j]>Qmax)
	  Qmax = Q[i][j];
      }else if(phi_r[i][j]==phi_tilde[i][j]) {
	Q[i][j] = 1.0;
      }else{
	Q[i][j] = 0.0;
      }
  printf("Qmax = %e\n",Qmax);
}

void Lucy::update_psi_r(void)
{
  /* loop over the reconstructed image */
  /*for(int ia=0;ia<nx;ia++)
    for(int jb=0;jb<ny;jb++)
      psi_r[ia][jb] *= Q[ia][jb];
  */
  
  Real **pconv = cuda_convolve_2d(Q,A,nx,ny);


  /* loop over x and y */
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
    {

      /* reset psi_r */
      psi_r[i][j] *= pconv[i][j];

    }

  /* free memory */
  deallocate_2d_array(pconv,nx,ny);

  return;
}

int Lucy::find_i_min(int ia, int jb)
{
  return 0;
}
int Lucy::find_i_max(int ia, int jb)
{
  return 0;
}
int Lucy::find_j_min(int ia, int jb)
{
  return 0;
}
int Lucy::find_j_max(int ia, int jb)
{
  return 0;
}
int Lucy::find_ia_min(int i, int j)
{
  int ia_min = 0;
  if(i < na-1)
    ia_min = na-1-i;
  return ia_min;
}
int Lucy::find_ia_max(int i, int j)
{
  int ia_max = 0;
  if(i > nx-1 - 2*na)
    ia_max = na-1-i;
  return ia_max;
}
int Lucy::find_jb_min(int i, int j)
{
  int jb_min = 0;
  if(j < nb-1)
    jb_min = nb-1-j;
  return jb_min;
}
int Lucy::find_jb_max(int i, int j)
{
  int jb_max = 0;
  if(j > ny-1 - 2*nb)
    jb_max = nb-1-j;
  return jb_max;
}
