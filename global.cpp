#include"global.h"
Real **allocate_2d_array(int n, int l)
{
        Real **x;

        x = new Real *[n];
        for(int i=0;i<n;i++)
                x[i] = new Real[l];

        for(int i=0;i<n;i++)
                for(int j=0;j<l;j++)
                        x[i][j] = 0.0;
        return x;
}
void deallocate_2d_array(Real **x, int n, int l)
{
  for(int i=0;i<n;i++)
    delete[] x[i];
  delete x;
}
