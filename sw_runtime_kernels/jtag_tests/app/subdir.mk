#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

subdirs-lnn-y   += shared modules leon
ccopt-lnn-y   += -falign-functions=64 -falign-loops=64

CURRENT_DIR := $(abspath ./)
FIRMWARE_VPU_ABS_DIR := $(abspath ${FIRMWARE_VPU_DIR})

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
REL_TO_ROOT := $(subst /, ,${CURRENT_DIR})
REL_TO_ROOT := $(patsubst %,../,${REL_TO_ROOT})
REL_TO_ROOT := $(subst $(SPACE),,$(REL_TO_ROOT))
FIRMWARE_VPU_REL_THROUGH_ROOT := $(REL_TO_ROOT)$(FIRMWARE_VPU_ABS_DIR)
VSYSTEM := $(FIRMWARE_VPU_REL_THROUGH_ROOT)/system

ccopt-lnn-y += -DCONFIG_USE_COMPONENT_NN
ccopt-shave_nn-y += -DCONFIG_USE_COMPONENT_NN

# Needed for nn_shave_manager single shave
ccopt-lnn-y += -DCONFIG_ACTSHV_START_STOP_PERF_TEST

subdirs-lnn-y += nn/inference_runtime_common ../../kernels
subdirs-shave_nn-y += ../../kernels nn/inference_runtime_common

include-dirs-lnn-y += $(FIRMWARE_VPU_REL_THROUGH_ROOT)/drivers/resource/barrier/inc

ccopt-lnn-$(CONFIG_BUILD_RELEASE) += -DNDEBUG
ccopt-shave_nn-$(CONFIG_BUILD_RELEASE) += -DNDEBUG

global-symbols-y += lnn_text_start
