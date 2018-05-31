swig3.0 -c++ -python composition_interface.i

g++ -std=c++11 -Wall -shared -fpic composition_interface_wrap.cxx -L../build/lib -lcm -I/usr/include/python3.6 -I.. -o _composition_api.so

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../build/lib


python3 convolution_composition_test.py