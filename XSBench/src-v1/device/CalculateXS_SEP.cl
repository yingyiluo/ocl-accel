#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void calculate_macro_xs(int lookups, long n_isotopes, long n_gridpoints,
								constant int *restrict num_nucs,
								constant int *restrict mats,
								constant double *restrict concs,
								__global int *restrict energy_grid_xs,
								__global NuclideGridPoint *restrict nuclide_grids,
								__global ulong *restrict vhash)
{
	ulong vhash_result = 0;
	int nucs_idx[12];
	nucs_idx[0] = 0;
	for ( int i = 1; i < 12; i++ )
		nucs_idx[i] = nucs_idx[i-1] + num_nucs[i-1];
	
	for( int i = 0; i < lookups; i++ )
	{
		double macro_xs_vector[5] = {0};
		for( int j = 0; j < 5; j++)
			macro_xs_vector[j] = 0;

		LookupContext lc = read_channel_intel(LC_QUEUE);
		int num_iters = num_nucs[lc.mat];		
		NuclideGridPoint low[321];
		NuclideGridPoint high[321];
		for (int j = 0; j < num_iters; j++ )
		{
			int p_nuc = mats[nucs_idx[lc.mat] + j];
						
			long energy_at_nuc = (long) energy_grid_xs[lc.idx * n_isotopes + p_nuc]; // energy_grid_xs_bin[p_nuc];
			long nuclide_idx_at_nuc = p_nuc * n_gridpoints;
			bool ifend = (energy_at_nuc == n_gridpoints - 1);
			low[j] = ifend ? nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc - 1] : 
					nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc];
	
			high[j] = ifend ? nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc] : 
					nuclide_grids[nuclide_idx_at_nuc + energy_at_nuc + 1];
		}

		for( int j = 0; j < num_iters; j++ )
		{
			double xs_vector[5];

			// calculate the re-useable interpolation factor
			double f = (high[j].energy - lc.energy) / (high[j].energy - low[j].energy);

			// Total XS
			xs_vector[0] = mad( -f, (high[j].total_xs - low[j].total_xs), high[j].total_xs );
	
			// Elastic XS
			xs_vector[1] = mad( -f, (high[j].elastic_xs - low[j].elastic_xs), high[j].elastic_xs );
	
			// Absorbtion XS
			xs_vector[2] = mad( -f, (high[j].absorbtion_xs - low[j].absorbtion_xs), high[j].absorbtion_xs );
	
			// Fission XS
			xs_vector[3] = mad( -f, (high[j].fission_xs - low[j].fission_xs), high[j].fission_xs );
	
			// Nu Fission XS
			xs_vector[4] = mad( -f, (high[j].nu_fission_xs - low[j].nu_fission_xs), high[j].nu_fission_xs );
		
			double conc = concs[nucs_idx[lc.mat] + j];
			#pragma unroll
			for( int k = 0; k < 5; k++ )
				macro_xs_vector[k] += xs_vector[k] * conc;
		}
		//#ifdef VERIFICATION
		unsigned int hash = 5381;	
		hash = ((hash << 5) + hash) + (int)lc.energy;
		hash = ((hash << 5) + hash) + (int)lc.mat;
		#pragma unroll
		for(int k = 0; k < 5; k++)
			hash = ((hash << 5) + hash) + macro_xs_vector[k];
		vhash_result += hash % 1000;
		//#endif 
	}
	*vhash = vhash_result;
}
