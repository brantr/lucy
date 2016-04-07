/*! \file rng.c
 *  \brief Function definitions for easy random number generation */
#include"rng.h"
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>

/*! \var int flag_rng_gaussian
 *  \brief Initialization flag for Gaussian RNG */
int flag_rng_gaussian;
/*! \var int flag_rng_uniform
 *  \brief Initialization flag for uniform RNG */
int flag_rng_uniform;

/*! \var int flag_rng_direction
 *  \brief Initialization flag for direction RNG */
int flag_rng_direction;

/*! \var int flag_rng_integer
 *  \brief Initialization flag for direction RNG */
int flag_rng_integer;

/*! \var int flag_rng_permutation
 *  \brief Initialization flag for permutation RNG*/
int flag_rng_permutation;


/*! \var const gsl_rng_type *T_rng_gaussian
 * \brief RNG type for Gaussian PDF */
const gsl_rng_type *T_rng_gaussian;

/*! \var const gsl_rng_type *T_rng_uniform;
 * \brief RNG type for uniform PDF */
const gsl_rng_type *T_rng_uniform;

/*! \var const gsl_rng_type *T_rng_direction;
 * \brief RNG type for direction PDF */
const gsl_rng_type *T_rng_direction;

/*! \var const gsl_rng_type *T_rng_integer;
 * \brief RNG type for integer PDF */
const gsl_rng_type *T_rng_integer;


/*! \var const gsl_rng_type *T_rng_permutation;
 * \brief RNG type for permutation PDF */
const gsl_rng_type *T_rng_permutation;

/*! \var gsl_rng *r_rng_gaussian
 *  \brief Gaussian RNG */
gsl_rng *r_rng_gaussian;
/*! \var gsl_rng *r_rng_uniform
 *  \brief Uniform RNG */
gsl_rng *r_rng_uniform;
/*! \var gsl_rng *r_rng_integer
 *  \brief Integer RNG */
gsl_rng *r_rng_integer;

/*! \var gsl_rng *r_rng_direction
 *  \brief Direction RNG */
gsl_rng *r_rng_direction;

/*! \var gsl_rng *r_rng_permutation
 *  \brief Permutation RNG */
gsl_rng *r_rng_permutation;

/*! \var gsl_permuation *p_rng_permutation
 *  \brief Permutation */
gsl_permutation *p_rng_permutation;

/*! \fn double rng_gaussian(double mu, double sigma)
 *  \brief Returns a Gaussianly-distributed random number with mean mu and dispersion sigma */
double rng_gaussian(double mu, double sigma)
{
	if(!flag_rng_gaussian)
		initialize_rng_gaussian();
	return gsl_ran_gaussian(r_rng_gaussian, sigma) + mu;
}

/*! \fn void initialize_rng_gaussian(void)
 *  \brief Initializes Gaussian random number generator */
void initialize_rng_gaussian(void)
{
	flag_rng_gaussian = 1;

	gsl_rng_env_setup();

	T_rng_gaussian = gsl_rng_default;
	r_rng_gaussian = gsl_rng_alloc(T_rng_gaussian);
	
}

/*! \fn double rng_uniform(double a, double b)
 *  \brief Returns a uniformly-distributed random number between a and b*/
double rng_uniform(double a, double b)
{
	if(!flag_rng_uniform)
		initialize_rng_uniform();
	return gsl_ran_flat(r_rng_uniform, a, b);
}

/*! \fn size_t *rng_permutation(int n)
 *  \brief Returns [0,n-1], randomly permutated*/
size_t *rng_permutation(int n)
{
	size_t *permutation;
	if(!flag_rng_permutation)
		initialize_rng_permutation(n);

	if(!flag_rng_permutation)
		if(n!=p_rng_permutation->size)
			initialize_rng_permutation(n);
	

	//permutation = calloc_size_t_array(n);

	//for(int i=0;i<n;i++)
	//	permutation[i] = i;

	//gsl_permute(p_rng_permutation, permutation, 1, n);

	//return permutation;

	gsl_ran_shuffle(r_rng_permutation,p_rng_permutation->data, n, sizeof(size_t));
	return gsl_permutation_data(p_rng_permutation);
}

/*! \fn void initialize_rng_permutation(void)
 *  \brief Initializes integer random permutation generator*/
void initialize_rng_permutation(int n)
{
	if(!flag_rng_permutation)
	{
		flag_rng_permutation = 1;
		gsl_rng_env_setup();
		T_rng_permutation = gsl_rng_taus;
		r_rng_permutation = gsl_rng_alloc(T_rng_permutation);

		p_rng_permutation = gsl_permutation_alloc(n);
	}else{
		if(n!=p_rng_permutation->size)
		{
			gsl_permutation_free(p_rng_permutation);
			p_rng_permutation = gsl_permutation_alloc(n);
		}
	}

	gsl_permutation_init(p_rng_permutation);
	gsl_ran_shuffle(r_rng_permutation,p_rng_permutation->data, n, sizeof(size_t));

}


/*! \fn int rng_integer(int n)
 *  \brief Returns a uniformly-distributed random number between a and b*/
int rng_integer(int n)
{
	if(!flag_rng_integer)
		initialize_rng_integer();
	return gsl_rng_uniform_int(r_rng_integer, n);
}


/*! \fn void initialize_rng_integer(void)
 *  \brief Initializes integer random number generator */
void initialize_rng_integer(void)
{
	flag_rng_integer = 1;

	gsl_rng_env_setup();

	T_rng_integer = gsl_rng_taus;
	r_rng_integer = gsl_rng_alloc(T_rng_integer);
}


/*! \fn void initialize_rng_uniform(void)
 *  \brief Initializes uniform random number generator */
void initialize_rng_uniform(void)
{
	flag_rng_uniform = 1;

	gsl_rng_env_setup();

	//T_rng_uniform = gsl_rng_default;
	T_rng_uniform = gsl_rng_taus;
	r_rng_uniform = gsl_rng_alloc(T_rng_uniform);
}

/*! \fn void initialize_rng_direction(void)
 *  \brief Initializes direction random number generator */
void initialize_rng_direction(void)
{
	flag_rng_direction = 1;

	gsl_rng_env_setup();

	//T_rng_direction = gsl_rng_default;
	T_rng_direction = gsl_rng_taus;
	r_rng_direction = gsl_rng_alloc(T_rng_direction);
}


/*! \fn void set_rng_uniform_seed(int seed)
 *  \brief Set uniform rng seed */
void set_rng_uniform_seed(int seed)
{
	if(!flag_rng_uniform)
		initialize_rng_uniform();
	gsl_rng_set(r_rng_uniform, seed);
}

void set_rng_integer_seed(int seed)
{
	if(!flag_rng_integer)
		initialize_rng_integer();
	gsl_rng_set(r_rng_integer, seed);
}


/*! \fn void set_rng_gaussian_seed(int seed)
 *  \brief Set gaussian rng seed */
void set_rng_gaussian_seed(int seed)
{
	if(!flag_rng_gaussian)
		initialize_rng_gaussian();
	gsl_rng_set(r_rng_gaussian, seed);
}

/*! \fn void set_rng_direction_seed(int seed)
 *  \brief Set direction rng seed */
void set_rng_direction_seed(int seed)
{
	if(!flag_rng_direction)
		initialize_rng_direction();
	gsl_rng_set(r_rng_direction, seed);
}

/*! \fn double *rng_direction(int ndim)
 *  \brief Get a random unit vector */
double *rng_direction(int ndim)
{
	double *x;
	if(ndim<2)
	{
		printf("ndim must be > 1\n");
		return NULL;
	}
	//x = calloc_double_array(ndim);
	if(!(x = (double *) calloc(ndim,sizeof(double))))
	{
		printf("Error allocating x of size %d\n",ndim);
		return NULL;
	}
	switch(ndim)
	{
		case 2: gsl_ran_dir_2d(r_rng_direction,&x[0],&x[1]);
			break;
		case 3: gsl_ran_dir_3d(r_rng_direction,&x[0],&x[1],&x[2]);
			break;
		default: gsl_ran_dir_nd(r_rng_direction,ndim,x);
	}
	return x;
}
