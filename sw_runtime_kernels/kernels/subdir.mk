include-dirs-lrt-y += inc prebuild
include-dirs-shave-$(CONFIG_TARGET_SOC_MA2490) += inc
include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += inc inc/3720
subdirs-lrt-y += common
subdirs-lnn-y += common
subdirs-shave-y += common
subdirs-shave_nn-y += common
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./2490/*.c*) $(wildcard ./2490*.asm)
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./*.c*) $(wildcard ./*.asm)
srcs-shave_nn-$(CONFIG_TARGET_SOC_3720) += 3720/dma_shave_nn.cpp
#srcs-shave_nn-$(CONFIG_TARGET_SOC_3720) += 3720/dma_shave_nn.cpp 3720/nnActEntry.cpp

