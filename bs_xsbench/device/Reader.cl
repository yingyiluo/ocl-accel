__attribute__((max_global_work_dim(0)))
__kernel void reader(int lookups, __global ulong *restrict checksum) {
  ulong cs = 0;
  for (int i = 0; i < lookups; i++) {
    SearchContext ct = read_channel_altera(SC_QUEUE[1]);
    cs ^= ct.ll; 
  }
  *checksum = cs;
}
