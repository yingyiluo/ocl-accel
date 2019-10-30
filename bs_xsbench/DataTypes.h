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

typedef struct __attribute__((packed)) __attribute__((aligned(8))){
  double energy;
  int xs_ptrs[355];
} GridPoint_Array;

typedef struct __attribute__((packed)) __attribute__((aligned(128))){
  int xs_ptrs[355];
} GridPointXS;

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


typedef union data {
long l;
double d;
} l_to_d;

typedef struct __attribute__((aligned(16))) { 
  l_to_d data; 
  long index; 
} BSCacheUnion;

typedef struct __attribute__((packed)) __attribute((aligned(32))) {
  double energy;
  long idx;
  int mat;
} LookupContext;

#endif
