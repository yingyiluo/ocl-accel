#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "../DataTypes.h"

#define SIZE 12
#define STAGE_NUM 10
#define BUFFER_SIZE 32

channel SearchContext SC_QUEUE[2] __attribute__((depth(BUFFER_SIZE)));

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn(unsigned long * seed)
{
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (*seed) ) % m;
	*seed = n1;
	ret = (double) n1 / m;
	return ret;
}

// picks a material based on a probabilistic distribution
int pick_mat( unsigned long * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a 
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	// Also could be argued that doing fractions by weight would be 
	// a better approximation, but volume does a good enough job for now.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies
	
	double roll = rn(seed);
	double running = 0;
	bool ifmeet = false;
	int returni = 0;
	// makes a pick based on the distro
	for( int i = 1; i < 12; i++ )
	{
		running += dist[i];
		if( !ifmeet && (roll < running) ) {
			returni = i;
			ifmeet = true;
		}
	}

	return returni;
}

__attribute__((max_global_work_dim(0)))
__kernel void simulation(int lookups, 
			__constant BSCache *restrict bsc)
{
	long n_isotopes = 355;
	long n_gridpoints = 11303;
	long energy_grid_len = n_isotopes * n_gridpoints;
	// XS Lookup Loop
	// This loop is independent. Represents lookup events for many particles executed independently in one loop.
	//     i.e., All iterations can be processed in any order and are not related
	for( int i = 0; i < lookups; i++ )
	{
		// Particles are seeded by their particle ID
		unsigned long seed = ((unsigned long) i+ (unsigned long)1)* (unsigned long) 13371337;

		// Randomly pick an energy and material for the particle
		double p_energy = rn(&seed);
		int mat      = pick_mat(&seed); 

		int cll = 0;
    		int cul = 1023;
    		int cmid;
    		long ll = 0;
    		long ul = energy_grid_len - 1;
		#pragma unroll
		for (int n = 0; n < STAGE_NUM; n++) {
			cmid = cll + ((cul - cll) >> 1);
			BSCache cache_data = bsc[cmid];
			cul = (cache_data.data > p_energy) ? cmid : cul;
			cll = (cache_data.data > p_energy) ? cll : cmid;
			ul = (cache_data.data > p_energy) ? cache_data.index : ul;
			ll = (cache_data.data > p_energy) ? ll : cache_data.index;
		}

		SearchContext sc = {p_energy, ll, ul, mat};
    		write_channel_altera(SC_QUEUE[0], sc);
    }
}

#include "GridSearch_nochan.cl"
#include "CalculateXS_8_d16.cl"
