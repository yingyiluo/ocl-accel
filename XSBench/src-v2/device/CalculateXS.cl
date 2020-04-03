#include "Material.cl"
#include "concs.cl"

#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

channel ulong RESULT_QUEUE __attribute__((depth(1)));

__attribute__((max_global_work_dim(0)))
__kernel void calculate_macro_xs_one(
				__global const int *restrict energy_grid_xs,
				__global const double8 *restrict nuclide_grids,
				__global ulong *restrict vhash)
{
	ulong vhash_result = 0;

	long n_isotopes = 355;
	long n_gridpoints = 11303;
	for(int i = 0; i < lookups0; i++) {
		SearchContext lc = read_channel_altera(BS_QUEUE0);
		double lcenergy = lc.energy;
		long macro_xs_vector[5] = {0};
		for( int j = 0; j < 321; j++ )
		{
			double xs_vector[5];
			int p_nuc = mats0[j];
			double conc = concs[j].d;

			long energy_at_nuc = (long) energy_grid_xs[lc.ll * n_isotopes + p_nuc];
			long nu_idx = p_nuc * n_gridpoints + energy_at_nuc;
			double16 nu_data = *(__global double16 *) &nuclide_grids[nu_idx];

			double8 low = nu_data.lo;
			double8 high = nu_data.hi;

			// calculate the re-useable interpolation factor
			double f = (high.s0 - lcenergy) / (high.s0 - low.s0);

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
			#pragma unroll
			for( int k = 0; k < 5; k++ )
				macro_xs_vector[k] += (long) (xs_vector[k]);
		}
		#pragma unroll
		for(int k = 0; k < 5; k++)
			vhash_result += macro_xs_vector[k];
	}
	ulong r0 = read_channel_altera(RESULT_QUEUE);
	*vhash = r0 + vhash_result;
}

__attribute__((max_global_work_dim(0)))
__kernel void calculate_macro_xs_two(
				__global int *restrict energy_grid_xs,
				__global double8 *restrict nuclide_grids)
{
	ulong vhash_result = 0;

	long n_isotopes = 355;
	long n_gridpoints = 11303;
	for(long i = 0; i < lookups1; i++) {
		SearchContext lc = read_channel_altera(BS_QUEUE1);
		int lcmat = lc.mat;
		int iter_num = num_nucs[lcmat];
		int start_idx = cumulative_nucs[lcmat];
		double lcenergy = lc.energy;
		long macro_xs_vector[5] = {0};
		for( int j = 0; j < 27; j++ )
		{
			double xs_vector[5];
			if(i >= iter_num) {
				xs_vector[0] = 0;
			} else {
				int p_nuc = mats1[start_idx + j];
				double conc = concs[start_idx + j].d;

				long energy_at_nuc = (long) energy_grid_xs[lc.ll * n_isotopes + p_nuc]; // energy_grid_xs_bin[p_nuc];
				long nu_idx = p_nuc * n_gridpoints + energy_at_nuc;
				double16 nu_data = *(__global double16 *) &nuclide_grids[nu_idx];

				double8 low = nu_data.lo;
				double8 high = nu_data.hi;

				// calculate the re-useable interpolation factor
				double f = (high.s0 - lcenergy) / (high.s0 - low.s0);

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
				#pragma unroll
				for( int k = 0; k < 5; k++ )
					macro_xs_vector[k] += (long) (xs_vector[k]);	
			}
		}
		#pragma unroll
		for(int k = 0; k < 5; k++)
			vhash_result += macro_xs_vector[k];
	}
	write_channel_altera(RESULT_QUEUE, vhash_result);
}
