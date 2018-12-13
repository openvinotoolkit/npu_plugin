SRC := $(wildcard *.fbs)
OBJS := $(patsubst %.fbs, example/%_generated.h, $(wildcard *.fbs))
FLATC := flatc

example/%_generated.h: %.fbs
	#$(FLATC) -o example-py/ $< --python
	#$(FLATC) -o example-cpp/ $< --cpp
	$(FLATC) -o example-cpp-deserialize/ $< --cpp

all: clean $(OBJS) test debug


clean:
	-rm example-cpp/*_generated.h

test:
	#cd example-py && $(MAKE) all
	#cd example-cpp && $(MAKE) all
	cd example-cpp-deserialize && $(MAKE) all

debug:
	# $(FLATC) -o json/ graphfile.fbs --json -- example-cpp-deserialize/VPU3.cpp.bin
	#$(FLATC) -o json/ graphfile.fbs --json -- example-cpp/VPU3.cpp.bin
	#$(FLATC) -o json/ graphfile.fbs --json -- example-py/VPU3.py.bin

debug_full:
	# $(FLATC) -o json/ graphfile.fbs --json --defaults-json --unknown-json -- example-cpp-deserialize/VPU3.cpp.bin
	#$(FLATC) -o json/ graphfile.fbs --json --defaults-json --unknown-json -- example-cpp/VPU3.cpp.bin
	#$(FLATC) -o json/ graphfile.fbs --json --defaults-json --unknown-json -- example-py/VPU3.py.bin

flatbuffers:
	git clone https://github.com/google/flatbuffers.git
	cd flatbuffers;
	mkdir build; cd build; cmake ..; make -j 6;
