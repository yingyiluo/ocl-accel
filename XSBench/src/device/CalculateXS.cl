#include "Material.cl"
#include "concs.cl"

#define MASK 0x7

typedef struct __attribute__((packed)) __attribute__((aligned(64))) {
	double energy;
	int mat;
  	int iter_num;
	float xs_vector0;
	float xs_vector1;
	float xs_vector2;
	float xs_vector3;
	float xs_vector4;
} XS_Meta;

typedef struct __attribute__((packed)) __attribute__((aligned(64))) {
	double energy;
	double conc;
	long energy_addr;
	long nuclide_addr;
  	long total_iter;
	int sub_iter;
	int mat;
} ADDR_Meta;

typedef union i {
  int s[8];
  int8 v;
} unioni8;

channel XS_Meta XS_QUEUE __attribute__((depth(350)));
channel long RESULT_QUEUE __attribute__((depth(350)));
channel ADDR_Meta ADDR_QUEUE __attribute__((depth(350)));

__attribute__((max_global_work_dim(0)))
__kernel void addr_gen(int lookups)
{
	long n_isotopes = 360;
	long n_gridpoints = 11303;
	long total_iters = 1;
	for(int i = 0; i < lookups; i++) {
		SearchContext lc = read_channel_intel(SC_QUEUE[1]);
		int lcmat = lc.mat;
		int iter_num = num_nucs[lcmat];
		total_iters += iter_num;
		if(i == lookups - 1)
			total_iters -= 1;
		int start_idx = cumulative_nucs[lcmat];
		long energy_grid_idx = lc.ll * n_isotopes;
		for( int j = 0; j < iter_num; j++ ) {
			int p_nuc = mats[start_idx + j];
			double conc = concs[start_idx + j].d;
			ADDR_Meta addr = {lc.energy, conc, energy_grid_idx + p_nuc, p_nuc * n_gridpoints, total_iters, iter_num, lcmat};
			write_channel_intel(ADDR_QUEUE, addr);
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void calculate_macro_xs(__global int8 *restrict energy_grid_xs,
				__global double8 *restrict nuclide_grids)
{
	long n_gridpoints = 11303 - 1;
	long total_iter = 1;
	for(long i = 0; i < total_iter; i++) {
		ADDR_Meta addr = read_channel_intel(ADDR_QUEUE);
		total_iter = addr.total_iter;	
		int energy_addr = addr.energy_addr >> 3;
		unioni8 indices;
		indices.v = energy_grid_xs[energy_addr]; 
		long idx = (long) indices.s[addr.energy_addr & MASK];
		long nu_idx = idx + addr.nuclide_addr;
		double8 low, high;
		double16 nu_data = *(__global double16 *) &nuclide_grids[nu_idx];
		low = nu_data.lo;
		high = nu_data.hi;

		// calculate the re-useable interpolation factor
		double f = (high.s0 - addr.energy) / (high.s0 - low.s0);

		float xs_vector[5];
		double conc = addr.conc;
		// Total XS
		xs_vector[0] = mad( -f, (high.s1 - low.s1), high.s1 ) * conc;

		// Elastic XS
		xs_vector[1] = mad( -f, (high.s2 - low.s2), high.s2 ) * conc;

		// Absorbtion XS
		xs_vector[2] = mad( -f, (high.s3 - low.s3), high.s3 ) * conc;

		// Fission XS
		xs_vector[3] = mad( -f, (high.s4 - low.s4), high.s4 ) * conc;

		// Nu Fission XS
		xs_vector[4] = mad( -f, (high.s5 - low.s5), high.s5 ) * conc;
		XS_Meta meta = {addr.energy, addr.mat, addr.sub_iter, xs_vector[0], xs_vector[1], xs_vector[2], xs_vector[3], xs_vector[4]};
		write_channel_intel(XS_QUEUE, meta);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void accumulate_macro_xs(int lookups) {
	for (int i = 0; i < lookups; i++) {
		int j = 0;
		int iter_num;
		XS_Meta meta;
		float macro_xs_vector[5] = {0.0f};
		do {
			meta = read_channel_intel(XS_QUEUE);	
			iter_num = meta.iter_num;
			macro_xs_vector[0] += meta.xs_vector0;
			macro_xs_vector[1] += meta.xs_vector1;
			macro_xs_vector[2] += meta.xs_vector2;
			macro_xs_vector[3] += meta.xs_vector3;
			macro_xs_vector[4] += meta.xs_vector4;
			j++;
		} while(j < iter_num);

		ulong vhash_result = 0;
		unsigned int hash = 5381;	
		hash = ((hash << 5) + hash) + (int)meta.energy;
		hash = ((hash << 5) + hash) + (int)meta.mat;
		#pragma unroll
		for(int k = 0; k < 5; k++)
			hash = ((hash << 5) + hash) + macro_xs_vector[k];
		vhash_result = hash % 1000;

		write_channel_intel(RESULT_QUEUE, vhash_result);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void calculate_vhash(int lookups, __global ulong *restrict vhash) {
	ulong result = 0;
	for (int i = 0; i < lookups; i++)
		result += read_channel_intel(RESULT_QUEUE);
	*vhash = result; 
}
