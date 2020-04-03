#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<strings.h>
#include<math.h>
#include<omp.h>
#include<unistd.h>
#include<sys/time.h>
#include<assert.h>
#include "DataTypes.h"

// I/O Specifiers
#define INFO 1
#define DEBUG 1
#define SAVE 1

// Structures
typedef struct{
	double energy;
	int * xs_ptrs;
} GridPoint;

typedef struct{
	int nthreads;
	long n_isotopes;
	long n_gridpoints;
	int lookups;
	char * HM;
	int grid_type; // 0: Unionized Grid (default)    1: Nuclide Grid
	int hash_bins;
	int particles;
	int simulation_method;
} Inputs;

#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2

#define HISTORY_BASED 1
#define EVENT_BASED 2

// Function Prototypes
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(long a);

NuclideGridPoint ** gpmatrix(size_t m, size_t n);

void gpmatrix_free( NuclideGridPoint ** M );

int NGP_compare( const void * a, const void * b );

int ulong_compare( const void * a, const void * b );

void generate_grids( NuclideGridPoint ** nuclide_grids,
                     long n_isotopes, long n_gridpoints );
/*
void generate_grids_v( NuclideGridPoint ** nuclide_grids,
                     long n_isotopes, long n_gridpoints );
*/
void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, long n_isotopes,
                         long n_gridpoints );

GridPoint * generate_energy_grid( long n_isotopes, long n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids);

void initialization_do_not_profile_set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    long n_isotopes, long n_gridpoints );

void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints, GridPoint *energy_grid, NuclideGridPoint **nuclide_grids,
                           long idx, double *xs_vector, int grid_type, int hash_bins );
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int *num_nucs,
                         double **concs,
                         GridPoint *energy_grid,
                         NuclideGridPoint **nuclide_grids,
                         int **mats,
                         double *macro_xs_vector, int grid_type, int hash_bins );

/* 
// float
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints, GridPoint *energy_grid, NuclideGridPoint **nuclide_grids,
                           long idx, float *xs_vector, int grid_type, int hash_bins );
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int *num_nucs,
                         double **concs,
                         GridPoint *energy_grid,
                         NuclideGridPoint **nuclide_grids,
                         int **mats,
                         float *macro_xs_vector, int grid_type, int hash_bins );
*/

long grid_search( long n, double quarry, GridPoint * A);
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high);

int * load_num_nucs(long n_isotopes);
int ** load_mats( int * num_nucs, long n_isotopes );
double ** load_concs( int * num_nucs );
//double ** load_concs_v( int * num_nucs );
int pick_mat(unsigned long * seed);
double rn(unsigned long * seed);
int rn_int(unsigned long * seed);
void counter_stop( int * eventset, int num_papi_events );
void counter_init( int * eventset, int * num_papi_events );
void do_flops(void);
void do_loads( int nuc,
               NuclideGridPoint **nuclide_grids,
		       long n_gridpoints );	
Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
// double rn_v(void);
double round_double( double input );
unsigned int hash(char *str, int nbins);
size_t estimate_mem_usage( Inputs in );
void print_inputs(Inputs in, int nprocs, int version);
void print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash );
void binary_dump(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid, int grid_type);
void binary_read(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid, int grid_type);

GridPoint * generate_hash_table( NuclideGridPoint ** nuclide_grids,
                          long n_isotopes, long n_gridpoints, long M );

void initialization_do_not_profile_set_hash( GridPoint *energy_grid, NuclideGridPoint **nuclide_grids,
                    long n_isotopes, long n_gridpoints );

void run_event_based_simulation(Inputs in, GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids, int * num_nucs, int ** mats, double ** concs, int mype, unsigned long * vhash_result);

bool init();
void cleanup();
void run_simulation(Inputs in, double *energy, int *energy_grid_xs,
					GridPoint *energy_grid,
					NuclideGridPoint **nuclide_grids,
					int *num_nucs, int **mats, double **concs, 
					unsigned long *vhash_result);

void run_simulation_v2(Inputs in, GridPoint_Array *energy_grid_array,
          GridPoint *energy_grid,
          NuclideGridPoint **nuclide_grids,
          int *num_nucs, int **mats, double **concs, 
          unsigned long *vhash_result);

void run_simulation_v3(Inputs in, double *energy, GridPointXS *energy_grid_xs,
					GridPoint *energy_grid,
					NuclideGridPoint **nuclide_grids,
					int *num_nucs, int **mats, double **concs, 
					unsigned long *vhash_result);

void run_simulation_grid_sep(Inputs in, double *energy, GridPointXS *energy_grid_xs,
					GridPoint *energy_grid,
					NuclideGridPoint **nuclide_grids,
					int *num_nucs, int **mats, double **concs, 
					unsigned long *vhash_result);

void run_simulation_d16(Inputs in, double *energy, cl_int16 *energy_grid_xs,
		GridPoint *energy_grid,
		NuclideGridPoint **nuclide_grids, 
		int *num_nucs, int **mats, double **concs, 
		unsigned long *vhash);

void run_simulation_d8(Inputs in, double *energy, cl_int8 *energy_grid_xs,
		GridPoint *energy_grid,
		NuclideGridPoint **nuclide_grids, 
		int *num_nucs, int **mats, double **concs, 
		unsigned long *vhash);

void run_simulation_d4(Inputs in, double *energy, cl_int4 *energy_grid_xs,
		GridPoint *energy_grid,
		NuclideGridPoint **nuclide_grids, 
		int *num_nucs, int **mats, double **concs, 
		unsigned long *vhash);

void run_simulation_d2(Inputs in, double *energy, cl_int2 *energy_grid_xs,
		GridPoint *energy_grid,
		NuclideGridPoint **nuclide_grids, 
		int *num_nucs, int **mats, double **concs, 
		unsigned long *vhash);
#endif
