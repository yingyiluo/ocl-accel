#ifndef __DATATYPES_HEADER_H__
#define __DATATYPES_HEADER_H__

typedef struct __attribute__((packed)) __attribute__((aligned(64))) {
	double energy;
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

typedef struct __attribute__((packed)) __attribute__((aligned(32))) {
  double energy;
  long ll;
  long ul;
  int mat;
} SearchContext;

typedef struct __attribute__((aligned(16))) { 
  double data; 
  long index; 
} BSCache;

typedef struct __attribute__((packed)) __attribute((aligned(32))) {
  double energy;
  long idx;
  int mat;
} LookupContext;

#endif
