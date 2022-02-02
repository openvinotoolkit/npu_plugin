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

include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += inc
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += inc
include-dirs-shave_nn-y += inc

sys-nn-inference-runtime-common-components-srcs += $(wildcard src/*.c*)

srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
srcs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
srcs-lrt-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard src/2490/*.c*)
srcs-lrt-$(CONFIG_TARGET_SOC_3720) += $(wildcard src/37xx/*.c*)

