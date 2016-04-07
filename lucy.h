#ifndef LUCY_H
#define LUCY_H
#include"global.h"

class Lucy
{
  public:
    int r;  /*iteration*/
    int nx; /*x-size of image*/
    int ny; /*y-size of image*/
    int na; /*x-size of kernel*/
    int nb; /*y-size of kernel*/
    Real **phi_tilde; /* observed image */
    Real **phi_r;	/* approximation to image */
    Real **psi_r;	/* reconstructed image */
    Real **psi_0;	/* initial guess */
    Real **prior_r;	/* prior image */
    Real **Q;		/* intermediate kernel */
    Real **A;		/* PSF */

    Lucy(void);
    void LoadImage(Real **phi_in, int nx_in, int ny_in);
    void LoadPriorImage(Real **prior_in, int nx_in, int ny_in);
    void LoadKernel(Real **A_in, int na_in, int nb_in);
    void SaveInitialImage(void);
    void UpdateEstimate(void);
    void calculate_phi_r(void);
    void calculate_Q_r(void);
    void update_psi_r(void);
    int find_ia_min(int i, int j);
    int find_ia_max(int i, int j);
    int find_jb_min(int i, int j);
    int find_jb_max(int i, int j);
    int find_i_min(int ia, int jb);
    int find_i_max(int ia, int jb);
    int find_j_min(int ia, int jb);
    int find_j_max(int ia, int jb);

    void Deconvolve(int n_iter);
    void DeconvolveWithPrior(int n_iter);
};

#endif /* LUCY_H */
