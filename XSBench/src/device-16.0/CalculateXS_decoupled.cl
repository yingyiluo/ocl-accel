#include "Material.cl"
#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

channel XS_Meta XS_QUEUE __attribute__((depth(350)));
channel long RESULT_QUEUE __attribute__((depth(350)));

__attribute__((max_global_work_dim(0)))
__kernel void calculate_macro_xs(int lookups,
				__constant double *restrict concs,
				__global int *restrict energy_grid_xs,
				__global NuclideGridPoint *restrict nuclide_grids)
{
	long n_isotopes = 355;
	long n_gridpoints = 11303;
	for(int i = 0; i < lookups; i++) {
		SearchContext lc = read_channel_altera(SC_QUEUE[SIZE]);
		int lcmat = lc.mat;
		int iter_num = num_nucs[lcmat];
		int start_idx = cumulative_nucs[lcmat];
		double lcenergy = lc.energy;
		for( int j = 0; j < iter_num; j++ )
		{
			float xs_vector[5];
			int p_nuc = mats[start_idx + j];
			double conc = concs[start_idx + j];

			NuclideGridPoint low, high;
			long energy_at_nuc = (long) energy_grid_xs[lc.ll * n_isotopes + p_nuc]; // energy_grid_xs_bin[p_nuc];
			long nuclide_idx_at_nuc = p_nuc * n_gridpoints;
			low = (energy_at_nuc == n_gridpoints - 1) ? nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc - 1] : 
					nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc];

			high = (energy_at_nuc == n_gridpoints - 1) ? nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc] : 
					nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc + 1];

			// calculate the re-useable interpolation factor
			double f = (high.energy - lcenergy) / (high.energy - low.energy);

			// Total XS
			xs_vector[0] = mad( -f, (high.total_xs - low.total_xs), high.total_xs ) * conc;

			// Elastic XS
			xs_vector[1] = mad( -f, (high.elastic_xs - low.elastic_xs), high.elastic_xs ) * conc;

			// Absorbtion XS
			xs_vector[2] = mad( -f, (high.absorbtion_xs - low.absorbtion_xs), high.absorbtion_xs ) * conc;

			// Fission XS
			xs_vector[3] = mad( -f, (high.fission_xs - low.fission_xs), high.fission_xs ) * conc;

			// Nu Fission XS
			xs_vector[4] = mad( -f, (high.nu_fission_xs - low.nu_fission_xs), high.nu_fission_xs ) * conc;
			XS_Meta meta = {lcenergy, lcmat, iter_num, xs_vector[0], xs_vector[1], xs_vector[2], xs_vector[3], xs_vector[4]};
			write_channel_altera(XS_QUEUE, meta);
		}
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
			meta = read_channel_altera(XS_QUEUE);	
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

		write_channel_altera(RESULT_QUEUE, vhash_result);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void calculate_vhash(int lookups, __global ulong *restrict vhash) {
	ulong result = 0;
	for (int i = 0; i < lookups; i++)
		result += read_channel_altera(RESULT_QUEUE);
	*vhash = result; 
}
