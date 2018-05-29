swig3.0 -c++ -python composition_interface.i

g++ -shared -fpic composition_interface_wrap.cxx  -I/usr/include/python3.6 -I../contrib/googletest/googletest/include/ -I../ -o _composition_api.so

python3 convolution_composition_test.py