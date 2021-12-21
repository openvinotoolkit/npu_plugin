# {% copyright %}

include-dirs-shave_nn-y += inc

srcs-shave_nn-y += $(wildcard src/*.c*) $(wildcard src/*.asm)

# CONFIG_TARGET_SOC_* options are mutually exclusive. Only one can be enabled at a time
target-soc-37xx = $(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720)
srcs-shave_nn-$(target-soc-37xx) += $(wildcard src/37xx/*.c*) $(wildcard src/37xx/*.asm)

global-symbols-y += \
	shvNN0_nnActEntry
