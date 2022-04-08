#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

include-dirs-lnn-y += inc
include-dirs-shave-$(CONFIG_TARGET_SOC_MA2490) += inc
include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += inc inc/3720
include-dirs-lnn-$(CONFIG_TARGET_SOC_3720) += inc/3720

srcs-lnn-$(CONFIG_TARGET_SOC_3720) += 3720/dma_shave_nn.cpp
