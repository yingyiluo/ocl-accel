#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define M 1000

ulong xorshift128plus(const ulong2* s)
{
        ulong x = (*s).x;
        ulong const y = (*s).y;
        (*s).x = y;
        x ^= x << 23; // a
        (*s).y = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
        return (*s).y + y;
}

long search_data(double data,
                 const long len,
                 __global double *restrict data_array) {
  long ll = 0;
  long ul = len - 1;
  long interval = ul - ll;
  long mid;
  while (interval > 1) {
    mid = ll + ( interval / 2);
    long d = data_array[mid];
    if(d > data)
      ul = mid;
    else
      ll = mid;
    interval = ul - ll;
  }
  return ll;
}

__kernel void bsearch(const ulong2 seed,
		      const long len, 
                      __global double *restrict data_array, 
                      __global ulong *restrict checksum) {
  uint i;
  ulong cs = 0;
  double target;
  for(i = 0; i < M; i++) {
    target = (double) (xorshift128plus(&seed) % len); //maxValue = last data + 1
    // printf("i %d, target: %f\n", i, target);
    long pos = search_data(target, len, data_array);
    cs ^= pos;
  }
  *checksum = cs;
}
