#include "XSbench_header.h"

void run_event_based_simulation(Inputs in, GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids, int * num_nucs, int ** mats, double ** concs, int mype, unsigned long * vhash_result)
{
	if( mype == 0)	
		printf("Beginning event based simulation...\n");

	unsigned long vhash = 0;
		// Initialize parallel PAPI counters
		#ifdef PAPI
		int eventset = PAPI_NULL; 
		int num_papi_events;
		#pragma omp critical
		{
			counter_init(&eventset, &num_papi_events);
		}
		#endif

		// Initialize RNG seeds for threads
		int thread = omp_get_thread_num();

		// XS Lookup Loop
		// This loop is independent. Represents lookup events for many particles executed independently in one loop.
		//     i.e., All iterations can be processed in any order and are not related
		#pragma omp for schedule(guided)
		for( int i = 0; i < in.lookups; i++ )
		{
			// Status text
			if( INFO && mype == 0 && thread == 0 && i % 2000 == 0 )
				printf("\rCalculating XS's... (%.0lf%% completed)",
						(i / ( (double) in.lookups / (double) in.nthreads ))
						/ (double) in.nthreads * 100.0);
			// Particles are seeded by their particle ID
			unsigned long seed = ((unsigned long) i+ (unsigned long)1)* (unsigned long) 13371337;

			// Randomly pick an energy and material for the particle
			double p_energy = rn(&seed);
			int mat      = pick_mat(&seed); 

			long idx = grid_search( in.n_isotopes * in.n_gridpoints, p_energy,
	    	               energy_grid);	

			// Verification hash calculation
			// This method provides a consistent hash accross
			// architectures and compilers.
			#ifdef VERIFICATION
			vhash ^= idx;	
			// debugging
			//printf("HOST: E = %lf idx = %ld\n", p_energy, idx);
			#endif
	}
	*vhash_result = vhash;
}
