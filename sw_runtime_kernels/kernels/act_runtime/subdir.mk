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

include-dirs-shave_nn-y += inc

srcs-shave_nn-y += $(wildcard src/*.c*) $(wildcard src/*.asm)

# CONFIG_TARGET_SOC_* options are mutually exclusive. Only one can be enabled at a time
target-soc-37xx = $(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720)
srcs-shave_nn-$(target-soc-37xx) += $(wildcard src/37xx/*.c*) $(wildcard src/37xx/*.asm)
