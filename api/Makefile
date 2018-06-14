PYINC := $(shell python3-config --includes)


all:
	swig3.0 -c++ -python composition_interface.i
	g++ -std=c++11 -Wall -shared -fpic composition_interface_wrap.cxx -L../build/lib -lcm  $(PYINC) -I.. -o _composition_api.so
	LD_LIBRARY_PATH=../build/lib/ python3 convolution_composition_test.py

clean:
	rm *.cxx || true
	rm -r python_unittests_xml || true
	rm *.so || true
	rm *.blob || true
	rm composition_api.py || true
	rm -rf __pycache__ || true
