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

typedef struct __attribute__((packed)) __attribute((aligned(32))) cb {
  context buffer[250];
  char head;
  char tail;
  char size;
  bool full;
} circular_buffer;

#define M 1000

channel context CONTEXT_QUEUE __attribute__((depth(32)));

void init_cbuf(__local circular_buffer *restrict cbuf) {
  cbuf->head = 0;
  cbuf->tail = 0;
  cbuf->size = 250;
  cbuf->full = false;
}

bool push(__local circular_buffer *restrict cbuf, context ct) {
  if(cbuf->full)
    return false;
  cbuf->buffer[cbuf->head] = ct;
  cbuf->head = (cbuf->head + 1) % cbuf->size;
  cbuf->full = (cbuf->head == cbuf->tail);
  return true;
}

context pop(__local circular_buffer *restrict cbuf, bool *v) {
  if(!cbuf->full && (cbuf->head == cbuf->tail)) {
    *v = false;
    return (context){};
  }
  context c = cbuf->buffer[cbuf->tail];
  cbuf->tail = (cbuf->tail + 1) % cbuf->size;
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
  uint i = 0;
  ulong cs = 0;
  result rs[M];
  __local circular_buffer cbuf;
  init_cbuf(&cbuf);
  while (i < M) {
    bool valid1;
    context ct = pop(&cbuf, &valid1); 
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
      i++;
      //printf("inc i %d, d: %f, p: %ld\n", i, ct.data, ct.ll);
      continue;
    }
    long d = data_array[mid];
    // mem_fence(CLK_GLOBAL_MEM_FENCE);
    if(d > ct.data)
      ct.ul = mid;
    else
      ct.ll = mid;
    while(!push(&cbuf, ct)){} 
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
