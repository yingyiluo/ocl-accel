#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

channel LookupContext LC_QUEUE __attribute__((depth(32)));
#define BANK_SIZE 8
#define BUFFER_SIZE 512

bool push(__local SearchContext *restrict cbuf, ushort4 *restrict cbuf_meta, SearchContext ct) {
  if((*cbuf_meta).z)
    return false;
  cbuf[((*cbuf_meta).x & 0x1ff)] = ct;
  (*cbuf_meta).x = ((*cbuf_meta).x + 1) % BUFFER_SIZE;
  (*cbuf_meta).z = ((*cbuf_meta).x == (*cbuf_meta).y) ? 1 : 0;
  return true;
}

SearchContext pop(__local SearchContext *restrict cbuf, ushort4 *restrict cbuf_meta, bool *v) {
  *v = !(!(*cbuf_meta).z && ((*cbuf_meta).x == (*cbuf_meta).y));
  SearchContext c = cbuf[((*cbuf_meta).y & 0x1ff)];
  ushort inc = *v ? 1 : 0;
  (*cbuf_meta).y = ((*cbuf_meta).y + inc) % BUFFER_SIZE;
  return c;
}

__kernel void grid_search(int lookups, __global double *restrict A) {
  uint i = 0;
  __local SearchContext cbuf[BANK_SIZE][BUFFER_SIZE]  __attribute__((doublepump, bank_bits(9, 10, 11), bankwidth(32)));;
  ushort4 cbuf_meta[BANK_SIZE]; 
  #pragma unroll
  for (int n = 0; n < BANK_SIZE; n++)
    cbuf_meta[n] = (ushort4) (0, 0, 0, 512);

  while (i < lookups) {
    #pragma unroll
    for(int n = 0; n < BANK_SIZE; n++) {
      bool valid1;
      SearchContext sc = pop(cbuf[n], &(cbuf_meta[n]), &valid1);
      if(!valid1) {
        bool valid2;
        sc = read_channel_nb_intel(SC_QUEUE, &valid2);
        if(!valid2)
          continue;
      }
      long interval = sc.ul - sc.ll;
      long mid = sc.ll + ((sc.ul - sc.ll) >> 1);
      if(interval <= 1) {
        LookupContext lc = {sc.energy, sc.ll, sc.mat};
        write_channel_intel(LC_QUEUE, lc);
        i++;
        //printf("inc i %d, d: %f, p: %ld\n", i, ct.data, ct.ll);
        continue;
      }
      double d = A[mid];
      sc.ul = (d > sc.energy) ? mid : sc.ul;
      sc.ll = (d > sc.energy) ? sc.ll : mid;

      push(cbuf[n], &(cbuf_meta[n]), sc);
    }
  }
}
