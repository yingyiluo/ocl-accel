#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__attribute__((max_global_work_dim(0)))
__kernel void grid_search(int lookups, __global double *restrict A) {
  for (int i = 0; i < lookups; i++) {
    long mid, ul, ll;
    SearchContext ct = read_channel_altera(SC_QUEUE[0]);
    ul = ct.ul;
    ll = ct.ll;
    for (int cid = 0; cid < SIZE; cid++) {
      mid = ll + ((ul - ll) >> 1);
      double d = A[mid];
      ul = (d > ct.energy) ? mid : ul;
      ll = (d > ct.energy) ? ll : mid;
    }
    SearchContext ct_new = {ct.energy, ll, 0, ct.mat};
    write_channel_altera(SC_QUEUE[1], ct_new); 
  }
}
