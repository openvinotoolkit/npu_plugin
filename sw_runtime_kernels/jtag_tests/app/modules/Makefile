SCHEMA_PATH := ${VPUIP_2_Directory}/system/nn/blob/2490/schema
SRC := $(wildcard *.fbs)
BUILDDIR ?= .
OBJS := $(patsubst $(SCHEMA_PATH)/src/schema/%.fbs, $(BUILDDIR)/include/%_generated.h, $(wildcard $(SCHEMA_PATH)/src/schema/*.fbs))
FLATC := $(SCHEMA_PATH)/flatbuffers/flatc


$(BUILDDIR)/include/%_generated.h: $(SCHEMA_PATH)/src/schema/%.fbs
	@echo "MDKBUILD_DIR=" $(BUILDDIR)
	$(FLATC) -o $(BUILDDIR)/include/ $< --cpp --gen-object-api

all:
	@echo "Make Options: "
	@echo "- make schema"

clean:
	-rm $(BUILDDIR)/include/*_generated.h -f

$(FLATC):
	cd $(SCHEMA_PATH)/flatbuffers; cmake -DFLATBUFFERS_BUILD_TESTS=OFF .
	cd $(SCHEMA_PATH); $(MAKE) -C flatbuffers -j

schema: $(FLATC)
	$(MAKE) $(OBJS)

.PHONY: clean schema
