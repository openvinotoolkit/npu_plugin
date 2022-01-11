include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += inc
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += inc
include-dirs-shave_nn-y += inc

sys-nn-inference-runtime-common-components-srcs += $(wildcard src/*.c*)

srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
srcs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
srcs-lrt-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard src/2490/*.c*)
srcs-lrt-$(CONFIG_TARGET_SOC_3720) += $(wildcard src/37xx/*.c*)

