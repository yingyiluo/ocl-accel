#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define N 50

channel double DATA; // __attribute__((depth(8)));

__kernel void data_in(__global double *data_in) {
  for(int i = 0; i < N*2; i++) {
    write_channel_intel(DATA, data_in[i]);
  }
}

__kernel void data_out(__global double *data_out) {
  for(int i = 0; i < N; i++) {
    data_out[i] = 2.0 * read_channel_intel(DATA);
  }
}
