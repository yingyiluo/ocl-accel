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

#define M 1024
#define BANK_SIZE 4
#define BUFFER_SIZE 256

channel context CONTEXT_QUEUE __attribute__((depth(32)));

bool push(__local context *restrict cbuf, ushort4 *restrict cbuf_meta, context ct) {
  if((*cbuf_meta).z)
    return false;
  cbuf[((*cbuf_meta).x & 0xff)] = ct;
  (*cbuf_meta).x = ((*cbuf_meta).x + 1) % BUFFER_SIZE;
  (*cbuf_meta).z = ((*cbuf_meta).x== (*cbuf_meta).y) ? 1 : 0;
  return true;
}

context pop(__local context *restrict cbuf, ushort4 *restrict cbuf_meta, bool *v) {
  if(!(*cbuf_meta).z && ((*cbuf_meta).x == (*cbuf_meta).y)) {
    *v = false;
    return (context){};
  }
  context c = cbuf[((*cbuf_meta).y & 0xff)];
  (*cbuf_meta).y = ((*cbuf_meta).y + 1) % BUFFER_SIZE;
  *v = true;
  return c;
}

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
  uint n = 0;
  ulong cs = 0;
  result rs[M];
  __local context cbuf[BANK_SIZE][BUFFER_SIZE]  __attribute__((bank_bits(9,8),bankwidth(32)));;
  // cbuf_meta 
  // .x: head
  // .y: tail
  // .z: full
  // .w: size
  ushort4 cbuf_meta[BANK_SIZE]; 
  #pragma unroll
  for (int i = 0; i < BANK_SIZE; i++)
    cbuf_meta[i] = (ushort4) (0, 0, 0, 256);
  while (n < M) {
    #pragma unroll
    for (int i = 0; i < BANK_SIZE; i++) {
      bool valid1;
      context ct = pop(cbuf[i], &(cbuf_meta[i]), &valid1);
      if(!valid1) {
        bool valid2;
        ct = read_channel_nb_intel(CONTEXT_QUEUE, &valid2);
        if(!valid2)
          continue;
      }
      long interval = ct.ul - ct.ll;
      long mid = ct.ll + ((ct.ul - ct.ll) / 2);
      if(interval <= 1) {
        rs[i] = (result) {ct.data, ct.ll};
        cs ^= ct.ll;
        n++;
        //printf("inc i %d, d: %f, p: %ld\n", i, ct.data, ct.ll);
        continue;
      }
      long d = data_array[mid];
      if(d > ct.data)
        ct.ul = mid;
      else
        ct.ll = mid;
      push(cbuf[i], &(cbuf_meta[i]), ct); 
    }
  }
  *checksum = cs;
}

__kernel void writer(const ulong2 seed,
		      const long len) {
  uint i;
  double target;
  for(i = 0; i < M; i++) {
    target = (double) (xorshift128plus(&seed) % len); //maxValue = last data + 1
    context ct = {target, 0, len - 1};
    write_channel_intel(CONTEXT_QUEUE, ct);         
  }
}
