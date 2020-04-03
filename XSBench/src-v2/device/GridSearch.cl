#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__attribute__((max_global_work_dim(0)))
__kernel void grid_search(int lookups, __global const double *restrict A) {
  for (int i = 0; i < lookups; i++) {
    long mid, ul, ll;
    SearchContext ct = read_channel_altera(SC_QUEUE);
    ul = ct.ul;
    ll = ct.ll;
    int mat = ct.mat;
    #pragma unroll 1
    for (int cid = 0; cid < SIZE; cid++) {
        mid = ll + ((ul - ll) >> 1);
        double d = A[mid];
        ul = (d > ct.energy) ? mid : ul;
        ll = (d > ct.energy) ? ll : mid;
    }
    SearchContext ct_new = {ct.energy, ll, 0, mat};
    if(mat == 0) {
        write_channel_altera(BS_QUEUE0, ct_new);
    } else {
        write_channel_altera(BS_QUEUE1, ct_new);
    }
  }
}
