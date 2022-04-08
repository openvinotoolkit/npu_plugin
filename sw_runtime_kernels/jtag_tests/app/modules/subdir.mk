RELPATH := $(NEXTGEN_INSTALL_DIR)/make/relpath.sh
OBJDIR := $(shell $(RELPATH) $(GET_CURRENT_DIR) $(BUILDDIR))

#referecne schema files in include folders
include-dirs-lnn-y += . $(OBJDIR)

subdirs-lnn-y   += 37xx
include-dirs-lnn-y += 37xx

srcs-lnn-y += $(wildcard *.cpp)
