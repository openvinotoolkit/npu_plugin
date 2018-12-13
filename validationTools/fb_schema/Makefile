SRC := $(wildcard *.fbs)
OBJS := $(patsubst %.fbs, example/%_generated.h, $(wildcard *.fbs))
FLATC := flatc

example/%_generated.h: %.fbs
	$(FLATC) -o blobcompareTool/ $< --cpp

all: clean $(OBJS) test


clean:
	-rm blobcompareTool/*_generated.h blobcompareTool/*.o 

test:
	cd blobcompareTool && $(MAKE) all
	
flatbuffers:
	git clone https://github.com/google/flatbuffers.git
	cd flatbuffers;
	mkdir build; cd build; cmake ..; make -j 6;
