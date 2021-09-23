RELPATH := $(NEXTGEN_INSTALL_DIR)/make/relpath.sh
OBJDIR := $(shell $(RELPATH) $(GET_CURRENT_DIR) $(BUILDDIR))

#referecne schema files in include folders
include-dirs-los-y += .
include-dirs-lrt-y += . $(OBJDIR)

ifeq ($(CONFIG_TARGET_SOC_3600)$(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720), y)
subdirs-los-y   += 37xx
subdirs-lrt-y   += 37xx
include-dirs-los-y += 37xx
include-dirs-lrt-y += 37xx
else
subdirs-los-y   += 2490
subdirs-lrt-y   += 2490
include-dirs-los-y += 2490
include-dirs-lrt-y += 2490
endif

srcs-los-y += $(wildcard *.cpp)
srcs-lrt-y += $(wildcard *.cpp)
srcs-lrt-y += ShaveElfMetadata/ShaveElfMetadataParser.cpp

