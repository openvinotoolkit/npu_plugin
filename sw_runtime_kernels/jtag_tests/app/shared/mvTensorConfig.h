// {% copyright %}
#ifndef MVTENSOR_CONFIG_H
#define MVTENSOR_CONFIG_H

// System resources configuration

static const int MVTENSOR_CACHE_LINE_LENGTH = 64;
static const int MVTENSOR_PARAM_SIZE = 512;
static const int MVTENSOR_HEAP_DATA_SIZE = MVTENSOR_CMX_BUFFER * 1024;
static const int MVTENSOR_STACK_SIZE = 1024;

#define MVTENSOR_MAX_SHAVES 12
#define MVTENSOR_MAX_NCES 0
#define MVTENSOR_MAX_CMX_SLICES 15
#define MVTENSOR_MUTEX 5
#define MV_TENSOR_DBG_MSG_SIZE 120

#ifndef __MOVICOMPILE__
#	ifndef DDR_DATA
#		define DDR_DATA  __attribute__((section(".ddr.data"), aligned(MVTENSOR_CACHE_LINE_LENGTH)))
#	endif
#       ifndef DDR_BSS
#       	define DDR_BSS __attribute__((section(".ddr.bss"), aligned(MVTENSOR_CACHE_LINE_LENGTH)))
#       endif
#	define NOCACHE __attribute__((section(".cmx.data"), aligned(MVTENSOR_CACHE_LINE_LENGTH)))
#else
#	ifndef SLICE_BSS
#		define SLICE_BSS __attribute__((section(".bss"), aligned(MVTENSOR_CACHE_LINE_LENGTH)))
#	endif
#endif

#endif
