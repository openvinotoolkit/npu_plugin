#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

include-dirs-shave_nn-y += $(OBJDIR)
$(info "objdir=$(OBJDIR)")

CURRENT_DIR := $(abspath ./)
FIRMWARE_VPU_ABS_DIR := $(abspath ${FIRMWARE_VPU_DIR})

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
REL_TO_ROOT := $(subst /, ,${CURRENT_DIR})
REL_TO_ROOT := $(patsubst %,../,${REL_TO_ROOT})
REL_TO_ROOT := $(subst $(SPACE),,$(REL_TO_ROOT))
FIRMWARE_VPU_REL_THROUGH_ROOT := $(REL_TO_ROOT)$(FIRMWARE_VPU_ABS_DIR)

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  leon/common leon/inference_runtime leon/common_runtime
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  leon
