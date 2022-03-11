#
# Copyright Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

include-dirs-lrt-y += inc prebuild
include-dirs-shave-$(CONFIG_TARGET_SOC_MA2490) += inc
include-dirs-shave-$(CONFIG_TARGET_SOC_3720) += inc inc/3720
include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += inc inc/3720
include-dirs-lrt-$(CONFIG_TARGET_SOC_3720) += inc/3720
subdirs-lrt-y += common
subdirs-lnn-y += common
subdirs-shave-y += common
#subdirs-shave-$(CONFIG_TARGET_SOC_MA2490) += common
subdirs-shave_nn-y += common
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./2490/*.c*) $(wildcard ./2490*.asm)
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./*.c*) $(wildcard ./*.asm)
srcs-lrt-$(CONFIG_TARGET_SOC_3720) += 3720/dma_shave_nn.cpp
