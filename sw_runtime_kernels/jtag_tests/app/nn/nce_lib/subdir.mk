include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += inc
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += inc

sys-nn-nce-lib-component-srcs-y += $(wildcard src/*.c*)

sys-nn-nce-lib-component-srcs-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard src/2490/*.c*)
sys-nn-nce-lib-component-srcs-$(CONFIG_TARGET_SOC_3100) += $(wildcard src/2490/*.c*)

# CONFIG_TARGET_SOC_* options are mutually exclusive. Only one can be enabled at a time
target-soc-37xx = $(CONFIG_TARGET_SOC_3600)$(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720)
sys-nn-nce-lib-component-srcs-$(target-soc-37xx) += $(wildcard src/3600/*.c*)

srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(sys-nn-nce-lib-component-srcs-y)
srcs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-nce-lib-component-srcs-y)
