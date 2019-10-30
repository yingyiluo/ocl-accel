#===============================================================================
# User Options
#===============================================================================

COMPILER    = gnu
OPTIMIZE    = no
DEBUG       = no
PROFILE     = no
MPI         = no
PAPI        = no
VEC_INFO    = no
VERIFY      = yes
BINARY_DUMP = no
BINARY_READ = no

ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

#===============================================================================
# Program name & source code list
#===============================================================================
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
INCS := $(wildcard *.h) 
INC_DIRS := ../common/inc
COMMON_SRC := $(wildcard ../common/src/AOCLUtils/*.cpp)
SRCS = \
Main.cpp \
io.c \
Simulation_float.c \
CalculateXS_float.c \
GridInit.c \
XSutils.cpp \
Materials.cpp
SRCS_NDR = \
Main_NDR_10bsc.cpp \
io.c \
Simulation_NDR.c \
CalculateXS_float.c \
GridInit.c \
XSutils.cpp \
Materials.cpp

TARGET := XSBench
TARGET_NDR := XSBench_NDR
TARGET_DIR := bin

LIB_DIRS :=

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
#CFLAGS := -std=gnu99 -Wall
CFLAGS :=  

# Linker Flags
LDFLAGS = -lm -lrt -lpthread

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  CC = g++
  CFLAGS += -fopenmp -flto
endif

# Intel Compiler
ifeq ($(COMPILER),intel)
  CC = icc
  CFLAGS += -qopenmp 
endif

# BG/Q gcc Cross-Compiler
ifeq ($(MACHINE),bluegene)
  CC = mpicc
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Compiler Vectorization (needs -O3 flag) information
ifeq ($(VEC_INFO),yes)
  CFLAGS += -ftree-vectorizer-verbose=6
endif

# PAPI source (you may need to provide -I and -L pointing
# to PAPI depending on your installation
ifeq ($(PAPI),yes)
  source += papi.c
  CFLAGS += -DPAPI
  #CFLAGS += -I/soft/apps/packages/papi/papi-5.1.1/include
  #LDFLAGS += -L/soft/apps/packages/papi/papi-5.1.1/lib -lpapi
  LDFLAGS += -lpapi
endif

# MPI
ifeq ($(MPI),yes)
  CC = mpicc
  CFLAGS += -DMPI
endif

# Verification of results mode
ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFICATION
endif

# Binary dump for file I/O based initialization
ifeq ($(BINARY_DUMP),yes)
  CFLAGS += -DBINARY_DUMP
endif

# Binary read for file I/O based initialization
ifeq ($(BINARY_READ),yes)
  CFLAGS += -DBINARY_READ
endif

#===============================================================================
# Targets to Build
#===============================================================================
all : $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(COMMON_SRC) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CC) $(CFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(COMMON_SRC) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LDFLAGS),$L) \
			-o $(TARGET_DIR)/$(TARGET)

ndr : $(TARGET_DIR)/$(TARGET_NDR)

$(TARGET_DIR)/$(TARGET_NDR) : Makefile $(SRCS_NDR) $(COMMON_SRC) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CC) $(CFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS_NDR) $(COMMON_SRC) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LDFLAGS),$L) \
			-o $(TARGET_DIR)/$(TARGET_NDR)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET) $(TARGET_DIR)/$(TARGET_NDR)

edit:
	vim -p $(SRCS) papi.c XSbench_header.h

bgqrun:
	qsub -t 10 -n 1 -O test XSBench
