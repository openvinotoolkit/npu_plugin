# {% copyright %}

kmb_or_tbh =

ifeq ($(CONFIG_TARGET_SOC_MA2490), y)
kmb_or_tbh = y
endif
ifeq ($(CONFIG_TARGET_SOC_3100), y)
kmb_or_tbh = y
endif

ifdef kmb_or_tbh
include-dirs-lrt-y += inc ../inc
include-dirs-lrt-y += inc/layers ../inc/layers

sys-nn-shave-lib-leon-srcs += $(wildcard src/2490/*.c*)
sys-nn-shave-lib-leon-srcs += $(wildcard src/2490/layers/parser_*.c*)

sys-nn-shave-lib-leon-srcs += $(wildcard src/2490/ShaveElfMetadata/*.c*)

srcs-lrt-$(CONFIG_TARGET_SOC_MA2490) += $(sys-nn-shave-lib-leon-srcs)

srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += src/pre_custom_cpp.cpp

ccopt-lrt-$(CONFIG_ENABLE_CUSTOM_KERNEL_PERF_COUNTERS) += -DENABLE_CUSTOM_KERNEL_PERF_COUNTERS
ccopt-lnn-$(CONFIG_ENABLE_CUSTOM_KERNEL_PERF_COUNTERS) += -DENABLE_CUSTOM_KERNEL_PERF_COUNTERS
ccopt-shave-$(CONFIG_ENABLE_CUSTOM_KERNEL_PERF_COUNTERS) += -DENABLE_CUSTOM_KERNEL_PERF_COUNTERS

ccopt-lrt-y += -DNN_PRE_ALPHA_UPA_SHAVE_TARGET=$(CONFIG_NN_PRE_ALPHA_UPA_SHAVE_TARGET)

ccopt-lrt-$(CONFIG_DEBUG_NN_SVU_RUNTIME) += -DDEBUG_NN_SVU_RUNTIME
ccopt-shave-$(CONFIG_DEBUG_NN_SVU_RUNTIME) += -DDEBUG_NN_SVU_RUNTIME

ccopt-lrt-$(CONFIG_SVU_STACK_USAGE_INSTRUMENTATION) += -DSVU_STACK_USAGE_INSTRUMENTATION
ccopt-shave-$(CONFIG_SVU_STACK_USAGE_INSTRUMENTATION) += -DSVU_STACK_USAGE_INSTRUMENTATION

# For RTEMS
ccopt-lrt-y += -DDISPATCHER_OS

# Only to get SHAVE specific defines for LRT code
subdirs-lrt-y += shave
subdirs-shave-y += shave
endif

# CONFIG_TARGET_SOC_* options are mutually exclusive. Only one can be enabled at a time
ifeq ($(CONFIG_TARGET_SOC_3600)$(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720), y)
$(info Using Shavelib Stub for 3600)
include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += inc
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += inc
srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(wildcard src/3600/*.c*)
#TODO: which functions need stubs?
endif
