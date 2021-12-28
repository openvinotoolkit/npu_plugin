# {% copyright %}

mv-tensor-defines-y                                  += -DMVTENSOR_CMX_BUFFER=$(CONFIG_MVTENSOR_CMX_BUFFER)
mv-tensor-defines-$(CONFIG_MVTENSOR_FAST_SVU)        += -DMV_TENSOR_FAST__OS_DRV_SVU
mv-tensor-defines-$(CONFIG_USE_COMPONENT_MEMMANAGER) += -DMVTENSOR_USE_MEMORY_MANAGER
mv-tensor-defines-$(CONFIG_MVTENSOR_L2C_COPY_BACK)   += -DIS_LEON_L2C_MODE_COPY_BACK

subdirs-los-y   += shared modules leon
subdirs-lrt-y   += shared modules leon

ccopt-los-y   += $(mv-tensor-defines-y)
ccopt-lrt-y   += $(mv-tensor-defines-y)

ccopt-los-y   += -falign-functions=64 -falign-loops=64
ccopt-lrt-y   += -falign-functions=64 -falign-loops=64

CURRENT_DIR := $(abspath ./)
VPUIP_2_ABS_DIR := $(abspath ${VPUIP_2_Directory})

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
REL_TO_ROOT := $(subst /, ,${CURRENT_DIR})
REL_TO_ROOT := $(patsubst %,../,${REL_TO_ROOT})
REL_TO_ROOT := $(subst $(SPACE),,$(REL_TO_ROOT))
VPUIP_2_REL_THROUGH_ROOT := $(REL_TO_ROOT)$(VPUIP_2_ABS_DIR)
VSYSTEM := $(VPUIP_2_REL_THROUGH_ROOT)/system

ccopt-lrt-y += -DCONFIG_USE_COMPONENT_NN
ccopt-lnn-y += -DCONFIG_USE_COMPONENT_NN
ccopt-shave-y += -DCONFIG_USE_COMPONENT_NN
ccopt-shave_nn-y += -DCONFIG_USE_COMPONENT_NN

subdirs-lrt-y += nn/common nn/nce_lib nn/shave_lib nn/inference_runtime_common ../../kernels
#subdirs-lrt-y += nn/common $(VSYSTEM)/nn/platform_abstraction $(VSYSTEM)/nn/blob nn/nce_lib nn/shave_lib nn/inference_runtime_common ../../kernels
subdirs-lnn-y += nn/common nn/inference_runtime_common
subdirs-shave-y += nn/common
subdirs-shave-y += nn/shave_lib
subdirs-shave-y += ../../kernels
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  act_shave_lib

subdirs-lnn-$(CONFIG_TARGET_SOC_3720) +=  act_shave_lib
subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += act_shave_lib
subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  ../../kernels

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += nn/common nn/inference_runtime_common

srcs-shave_nn-$(CONFIG_TARGET_SOC_3720) += nn/shave_lib/shave/src/pre_custom_cpp.cpp
include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += nn/shave_lib/inc nn/shave_lib/inc/layers

ccopt-lrt-$(CONFIG_NN_PROFILING) += -DNN_PROFILING
ccopt-lnn-$(CONFIG_NN_PROFILING) += -DNN_PROFILING
ccopt-shave_nn-$(CONFIG_NN_PROFILING) += -DNN_PROFILING

ccopt-lrt-$(CONFIG_NN_PROFILING_ALL) += -DNN_PROFILING_ALL
ccopt-lnn-$(CONFIG_NN_PROFILING_ALL) += -DNN_PROFILING_ALL
ccopt-shave_nn-$(CONFIG_NN_PROFILING_ALL) += -DNN_PROFILING_ALL

ccopt-lrt-$(CONFIG_NN_FATHOM_WORKAROUND_OUT_CHANNEL_OFFSET) += -DNN_FATHOM_WORKAROUND_OUT_CHANNEL_OFFSET
ccopt-lrt-$(CONFIG_NN_FATHOM_WORKAROUND_ODU_OFFSET) += -DNN_FATHOM_WORKAROUND_ODU_OFFSET

ccopt-lrt-$(CONFIG_NN_SPECIFY_DPU_MASK) += -DNN_DPU_MASK=$(CONFIG_NN_DPU_MASK)

ccopt-lrt-$(CONFIG_NN_USE_MEMORY_MANAGER) += -DNN_USE_MEMORY_MANAGER
ccopt-lnn-$(CONFIG_NN_USE_MEMORY_MANAGER) += -DNN_USE_MEMORY_MANAGER
ccopt-shave-$(CONFIG_NN_USE_MEMORY_MANAGER) += -DNN_USE_MEMORY_MANAGER
ccopt-shave_nn-$(CONFIG_NN_USE_MEMORY_MANAGER) += -DNN_USE_MEMORY_MANAGER

ccopt-shave_nn-$(CONFIG_NN_ENABLE_SHADOWING) += -DNN_ENABLE_SHADOWING
ccopt-lrt-$(CONFIG_NN_ENABLE_SPARSE_IDU_SHADOWING) += -DNN_ENABLE_SPARSE_IDU_SHADOWING
ccopt-lnn-$(CONFIG_NN_IR_VERBOSE_STALLS) += -DNN_IR_VERBOSE_STALLS
ccopt-lnn-$(CONFIG_NN_ENABLE_STACK_CHECKER) += -DNN_ENABLE_STACK_CHECKER
ccopt-lnn-$(CONFIG_NN_DUMP_INTERMEDIATE_BUFFERS) += -DNN_DUMP_INTERMEDIATE_BUFFERS
ccopt-lnn-$(CONFIG_NN_DESPARSIFY_RESULTS) += -DNN_DESPARSIFY_RESULTS
ccopt-lnn-$(CONFIG_NN_PRINT_DPU_REGISTERS) += -DNN_PRINT_DPU_REGISTERS
ccopt-lrt-$(CONFIG_NN_DMA_DRY_RUN) += -DNN_DMA_DRY_RUN
ccopt-lrt-$(CONFIG_NN_SAVE_DPU_REGISTERS) += -DNN_SAVE_DPU_REGISTERS
ccopt-shave_nn-$(CONFIG_NN_DPU_DRY_RUN) += -DNN_DPU_DRY_RUN
ccopt-shave_nn-$(CONFIG_NN_HW_STATS_PROF) += -DNN_HW_STATS_PROF
ccopt-shave_nn-$(CONFIG_NN_HW_DEBUG) += -DNN_HW_DEBUG
ccopt-shave_nn-y += -DNN_CM_CONV_MAX_CHANNELS=$(CONFIG_NN_CM_CONV_MAX_CHANNELS)

ccopt-lnn-$(CONFIG_BUILD_RELEASE) += -DNDEBUG
ccopt-shave_nn-$(CONFIG_BUILD_RELEASE) += -DNDEBUG

global-symbols-y += lnn_text_start

ccopt-lrt-$(CONFIG_NN_ENABLE_CONTEXT_DEBUGGING) += -DNN_ENABLE_CONTEXT_DEBUGGING
ccopt-lnn-$(CONFIG_NN_ENABLE_CONTEXT_DEBUGGING) += -DNN_ENABLE_CONTEXT_DEBUGGING
ccopt-shave_nn-$(CONFIG_NN_ENABLE_CONTEXT_DEBUGGING) += -DNN_ENABLE_CONTEXT_DEBUGGING

ccopt-lrt-$(CONFIG_NN_ENABLE_CONTEXT_SUPPORT) += -DNN_ENABLE_CONTEXT_SUPPORT
ccopt-lnn-$(CONFIG_NN_ENABLE_CONTEXT_SUPPORT) += -DNN_ENABLE_CONTEXT_SUPPORT
ccopt-shave_nn-$(CONFIG_NN_ENABLE_CONTEXT_SUPPORT) += -DNN_ENABLE_CONTEXT_SUPPORT

ccopt-lrt-y += -DNN_MAX_UPA_SHAVE_POOL_SIZE=$(CONFIG_NN_MAX_UPA_SHAVE_POOL_SIZE)
ccopt-shave-y += -DNN_MAX_UPA_SHAVE_POOL_SIZE=$(CONFIG_NN_MAX_UPA_SHAVE_POOL_SIZE)

ccopt-lrt-$(CONFIG_NN_ENABLE_SCALABILITY_REPORTING) += -DNN_ENABLE_SCALABILITY_REPORTING
ccopt-lnn-$(CONFIG_NN_ENABLE_SCALABILITY_REPORTING) += -DNN_ENABLE_SCALABILITY_REPORTING

ccopt-lrt-y += -DNN_SCALABILITY_REPORTING_PERIOD_MS=$(CONFIG_NN_SCALABILITY_REPORTING_PERIOD_MS)

srcs-lrt-$(CONFIG_TARGET_SOC_3720) += $(VPUIP_2_REL_THROUGH_ROOT)/drivers/nn/src/nn_fifo.cpp

$(info !!!!! subdirs-shave-y = !!!!!!!!!!! $(subdirs-shave-y))

