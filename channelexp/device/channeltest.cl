#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define N 50

channel double DATA_IN;

kernel void data_in(global double *data_in) {
  for(int i = 0; i < N; i++) {
    write_channel_intel(DATA_IN, data_in[i]);
  }
}

kernel void data_out(global double *data_out) {
  for(int i = 0; i < N; i++) {
    data_out[i] = 2.0 * read_channel_intel(DATA_IN);
  }
}
