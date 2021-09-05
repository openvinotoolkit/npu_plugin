/*
* {% copyright %}
*/
#pragma once

#define SHAVE_LIB_PARAM_SIZE 320
#define SHAVE_LIB_EXEC_CONTEXT_SIZE 1 * 1024
#define SHAVE_LIB_DATA_SIZE 112 * 1024

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)

#if defined(__leon_rt__)// || defined(__leon_nn__)
#include <OsDrvSvu.h>
#else
#include <mv_types.h>
#include <registersMyriad.h>
#define TOTAL_NUM_SHAVES CFG_NUM_SHAVES
#endif

// FIXME: pull from config
#define NN_CONTROLLER_SHAVE_TARGET 0

#define SYS_NUM_SHAVES TOTAL_NUM_SHAVES
#define MAX_SHAVE_ID (TOTAL_NUM_SHAVES - 1)
#define INVALID_SHAVE_ID (TOTAL_NUM_SHAVES)

//CMX_SLICE_SIZE

// Message queue from SVU to the runtime's LRT monitorMsgQueue()
#define SHAVE_LIB_SVU_LRT_MESSAGE_QUEUE_SIZE 16

// Minimum size for Shave heap is 1k
#define SHAVE_LIB_HEAP_SIZE 2 * 1024
#define SHAVE_LIB_HEAP_ALIGN 64

// TODO: find a reasonable size for this
#define SHAVE_LIB_CODE_MAX_SIZE 32 * 1024
#define SHAVE_LIB_STACK_MAX_SIZE 8 * 1024

// must be < 1. Only a single channel is created in nn_ipc.cpp
#define MSS_2_CTRL_SHV_ID 0x0

// data from dispatch layer to shave (8 bytes: void* kernel_entry + u32* params)
#define IPC_TO_SVU_TRANSFER_SIZE_U32 2
#define IPC_TO_SVU_KERNEL_ENTRY_OFFSET 0
#define IPC_TO_SVU_KR_PARAM_CMX_OFFSET 1

// data from shave to dispatch layer (4 bytes: u32 shaveID )
#define IPC_TO_LRT_TRANSFER_SIZE_U32 1
#define IPC_TO_LRT_SVU_ID_COMPLETE 0

#define SHAVE_WIN_A WINDOW_A
#define SHAVE_WIN_B WINDOW_B
#define SHAVE_WIN_C WINDOW_C
#define SHAVE_WIN_D WINDOW_D

#define SVU_WINDOW_A 0x1C000000
#define SVU_WINDOW_B 0x1D000000
#define WINDOWING_MASK 0x00FFFFFF

// Defined by the linker script
#define DATA_WINDOW_MASK SVU_WINDOW_A
#define CODE_WINDOW_MASK SVU_WINDOW_B

#endif // defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)
