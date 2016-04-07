#ifndef GLOBAL_H 
#define GLOBAL_H 


#if PRECISION == 1
#ifndef FLOAT_TYPEDEF_DEFINED
typedef float Real;
#endif //FLOAT_TYPEDEF_DEFINED
#endif //PRECISION == 1
#if PRECISION == 2
#ifndef FLOAT_TYPEDEF_DEFINED
typedef double Real;
#endif //FLOAT_TYPEDEF_DEFINED
#endif //PRECISION == 2

Real **allocate_2d_array(int n, int l);
void deallocate_2d_array(Real **x, int n, int l);


#endif /* GLOBAL_H */
