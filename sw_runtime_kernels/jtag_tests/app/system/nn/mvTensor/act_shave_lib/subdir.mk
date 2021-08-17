
include-dirs-shave_nn-y += $(OBJDIR)
$(warning "objdir=$(OBJDIR)")


include-dirs-shave_nn-y += leon/inc

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  shave_nn
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  leon

#presilicon-dir = $(MDK_ROOT_PATH)/presilicon

presilicon-dir = ../../../../../../../../thirdparty/vpuip_2/presilicon/

#presilicon-dir = ../../../../../../presilicon

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon

subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/drivers


shve-lib-dir = ../../../../../../system/nn/shave_lib
include-dirs-shave_nn-y += $(shve-lib-dir)/shave/inc
