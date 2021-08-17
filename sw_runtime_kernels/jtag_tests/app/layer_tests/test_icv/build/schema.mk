SCHEMA_PATH := ${VPUIP_2_Directory}/system/nn/blob/2490/schema
SRC := $(wildcard *.fbs)
BUILDDIR ?= .
OBJS := $(patsubst $(SCHEMA_PATH)/src/schema/%.fbs, $(BUILDDIR)/include/%_generated.h, $(wildcard $(SCHEMA_PATH)/src/schema/*.fbs))
FLATC := $(SCHEMA_PATH)/flatbuffers/flatc
USER_MAIN_MAKEFILE := $(firstword $(MAKEFILE_LIST))

$(BUILDDIR)/include/%_generated.h: $(SCHEMA_PATH)/src/schema/%.fbs
	echo "MDKBUILD_DIR=" $(BUILDDIR)
	$(FLATC) -o $(BUILDDIR)/include/ $< --cpp --gen-object-api

all:
	@echo "Make Options: "
	@echo "- make schema"
	@echo "OBJC $(OBJC)"
	@echo "FLATC $(FLATC)"
	@echo "SRC $(SRC)"
	@echo "SCHEMA_PATH $(SCHEMA_PATH)"
	@echo "(MAKE) $(MAKE)"
	@echo $<
	@echo "$(FLATC) -o $(BUILDDIR)/include/ $< --cpp --gen-object-api"
	@echo "$(MAKE) $(OBJS)"


clean:
	-rm $(BUILDDIR)/include/*_generated.h -f

$(FLATC):
	echo $(SCHEMA_PATH)
	cd $(SCHEMA_PATH)/flatbuffers; cmake -DFLATBUFFERS_BUILD_TESTS=OFF .
	cd $(SCHEMA_PATH); $(MAKE) -C flatbuffers -j

schema: $(FLATC)
	$(MAKE) -f $(USER_MAIN_MAKEFILE) $(OBJS)

.PHONY: clean schema
