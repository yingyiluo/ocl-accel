#include "../DataTypes.h"
#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__attribute__((max_global_work_dim(0)))
__kernel void grid_search(int lookups, __global double *restrict A) {
  int cid = 0; //get_compute_id(0);
  for (int i = 0; i < lookups; i++) {
    #pragma unroll
    for (cid = 0; cid < SIZE; cid++) {
      SearchContext ct = read_channel_altera(SC_QUEUE[cid]);
      long mid = ct.ll + ((ct.ul - ct.ll) >> 1);
      double d = A[mid];
      ct.ul = (d > ct.energy) ? mid : ct.ul;
      ct.ll = (d > ct.energy) ? ct.ll : mid;
      write_channel_altera(SC_QUEUE[cid+1], ct); 
    }
  }
}
