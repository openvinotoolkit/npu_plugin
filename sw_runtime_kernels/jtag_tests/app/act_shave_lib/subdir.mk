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
