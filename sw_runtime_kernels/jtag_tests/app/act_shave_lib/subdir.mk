include-dirs-shave_nn-y += $(OBJDIR)
$(info "objdir=$(OBJDIR)")

CURRENT_DIR := $(abspath ./)
VPUIP_2_ABS_DIR := $(abspath ${VPUIP_2_Directory})

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
REL_TO_ROOT := $(subst /, ,${CURRENT_DIR})
REL_TO_ROOT := $(patsubst %,../,${REL_TO_ROOT})
REL_TO_ROOT := $(subst $(SPACE),,$(REL_TO_ROOT))
VPUIP_2_REL_THROUGH_ROOT := $(REL_TO_ROOT)$(VPUIP_2_ABS_DIR)

#include-dirs-shave_nn-y += leon/inc
#include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(VPUIP_2_REL_THROUGH_ROOT)/system/nn/blob/2490/inc

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  leon/common leon/inference_runtime leon/common_runtime leon/act_runtime
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  leon

#subdirs-lrt-$(CONFIG_TARGET_SOC_3720) += $(VPUIP_2_REL_THROUGH_ROOT)/system/shave

#presilicon-dir := ../../../../../vpuip_2/presilicon
#presilicon-dir := $(VPUIP_2_REL_THROUGH_ROOT)/presilicon

#include-dirs-shave_nn-y += $(presilicon-dir)/swCommon/shave_code/include
#include-dirs-shave_nn-y += $(presilicon-dir)/drivers/shave/include

#include-dirs-lrt-y += $(presilicon-dir)/drivers/leon/drv/include
#include-dirs-lrt-y += $(presilicon-dir)/swCommon/leon/include
#include-dirs-lrt-y += $(presilicon-dir)/swCommon/shared/include

#subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon
#subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon

#subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/drivers

