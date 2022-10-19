RELPATH := $(NEXTGEN_INSTALL_DIR)/make/relpath.sh
OBJDIR := $(shell $(RELPATH) $(GET_CURRENT_DIR) $(BUILDDIR))

#referecne schema files in include folders
include-dirs-los-y += .
include-dirs-lrt-y += . $(OBJDIR)

subdirs-los-y   += 37xx
subdirs-lrt-y   += 37xx
include-dirs-los-y += 37xx
include-dirs-lrt-y += 37xx

srcs-los-y += $(wildcard *.cpp)
srcs-lrt-y += $(wildcard *.cpp)
srcs-lrt-y += ShaveElfMetadata/ShaveElfMetadataParser.cpp

