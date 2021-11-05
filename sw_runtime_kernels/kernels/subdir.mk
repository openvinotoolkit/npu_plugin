include-dirs-lrt-y += inc
include-dirs-shave-$(CONFIG_TARGET_SOC_MA2490) += inc
include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += inc inc/3720
subdirs-lrt-y += common
subdirs-lnn-y += common
subdirs-shave-y += common
subdirs-shave_nn-y += common
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./2490/*.c*) $(wildcard ./2490*.asm)
srcs-shave-y += $(wildcard ./*.c*) $(wildcard ./*.asm)
srcs-shave_nn-$(CONFIG_TARGET_SOC_3720) += $(wildcard ./*.c*) $(wildcard ./*.asm) $(wildcard ./3720/*.c*) $(wildcard ./3720/*.asm)

