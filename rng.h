#include<gsl/gsl_rng.h>
#include<gsl/gsl_permutation.h>
#ifndef BRANT_RNG
#define BRANT_RNG

extern int flag_rng_gaussian;
extern int flag_rng_uniform;
extern int flag_rng_direction;
extern int flag_rng_integer;
extern int flag_rng_permutation;

extern const gsl_rng_type *T_rng_gaussian;
extern const gsl_rng_type *T_rng_uniform;
extern const gsl_rng_type *T_rng_direction;
extern const gsl_rng_type *T_rng_integer;
extern const gsl_rng_type *T_rng_permutation;

extern gsl_rng *r_rng_gaussian;
extern gsl_rng *r_rng_uniform;
extern gsl_rng *r_rng_direction;
extern gsl_rng *r_rng_integer;
extern gsl_rng *r_rng_permutation;

extern gsl_permutation *p_rng_permutation;

/*! \fn double rng_gaussian(double mu, double sigma)
 *  \brief Returns a Gaussianly-distributed random number with mean mu and dispersion sigma */
double rng_gaussian(double mu, double sigma);

/*! \fn void initialize_rng_gaussian(void)
 *  \brief Initializes Gaussian random number generator */
void initialize_rng_gaussian(void);

/*! \fn int rng_uniform(int n)
 *  \brief Returns a random integer between 0 and n-1*/
int rng_integer(int n);

/*! \fn double rng_uniform(double a, double b)
 *  \brief Returns a uniformly-distributed random number between a and b*/
double rng_uniform(double a, double b);

/*! \fn double *rng_direction(int ndim)
 *  \brief Returns a random direction in n dimensions*/
double *rng_direction(int ndim);

/*! \fn size_t *rng_permutation(int N)
 *  \brief Returns [0,N-1] randomly permutated*/
size_t *rng_permutation(int N);


/*! \fn void initialize_rng_uniform(void)
 *  \brief Initializes uniform random number generator */
void initialize_rng_uniform(void);

/*! \fn void initialize_rng_direction(void)
 *  \brief Initializes direction random number generator */
void initialize_rng_direction(void);

/*! \fn void initialize_rng_integer(void)
 *  \brief Initializes integer random number generator */
void initialize_rng_integer(void);

/*! \fn void initialize_rng_permutation(void)
 *  \brief Initializes permutation generator */
void initialize_rng_permutation(int n);


/*! \fn void set_rng_integer_seed(int seed)
 *  \brief Set integer rng seed. */
void set_rng_integer_seed(int seed);

/*! \fn void set_rng_uniform_seed(int seed)
 *  \brief Set uniform rng seed. */
void set_rng_uniform_seed(int seed);
void set_rng_integer_seed(int seed);

/*! \fn void set_rng_gaussian_seed(int seed)
 *  \brief Set gaussian rng seed. */
void set_rng_gaussian_seed(int seed);

/*! \fn void set_rng_direction_seed(int seed)
 *  \brief Set direction rng seed. */
void set_rng_direction_seed(int seed);
#endif //BRANT_RNG
