#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct __attribute__((packed)) __attribute__((aligned(32))) c {
  double data;   //data might be other scalar type
  long ll;
  long ul;
} context;

typedef struct __attribute__((packed)) __attribute((aligned(16))) r {
  double data;
  long pos;
} result;

#define M 5

channel context CONTEXT_QUEUE __attribute__((depth(16)));

ulong xorshift128plus(const ulong2* s)
{
        ulong x = (*s).x;
        ulong const y = (*s).y;
        (*s).x = y;
        x ^= x << 23; // a
        (*s).y = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
        return (*s).y + y;
}

__kernel void bsearch(__global double *restrict data_array, 
          __global ulong *restrict checksum) {
  uint i = 0;
  ulong cs = 0;
  result rs[M];
  while (i < M) {
    context ct = read_channel_intel(CONTEXT_QUEUE);
    long interval = ct.ul - ct.ll;
    long mid = ct.ll + (ct.ul - ct.ll) >> 1;
    if(interval <= 1) {
      rs[i] = (result) {ct.data, ct.ll};
      cs ^= ct.ll;
      i++;
      printf("inc i, d: %f, p: %ld\n", ct.data, ct.ll);
      continue;
    }
    long d = data_array[mid];
    if(d > ct.data)
      ct.ul = mid;
    else
      ct.ll = mid;
    printf("ct: %f, %ld, %ld\n", ct.data, ct.ll, ct.ul);
    // Intel restriction: multiple kernels cannot write/read to the same channel simultaneously.
    write_channel_intel(CONTEXT_QUEUE, ct);
  }
  *checksum = cs;
}

__kernel void writer(const ulong2 seed,
		      const long len) {
  uint i;
  double target;
  for(i = 0; i < M; i++) {
    target = (double) (xorshift128plus(&seed) % len); //maxValue = last data + 1
    printf("target: %f\n", target);
    context ct = {target, 0, len - 1};
    write_channel_intel(CONTEXT_QUEUE, ct);         
  }
}
