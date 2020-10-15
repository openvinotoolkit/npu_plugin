# Instructions to Generate Blobs and Test Inputs

* Create a build directory and setup ``MCM_HOME`` env. variable.
 ```
 mkdir build
 ```
* Generate Make file
```
cd build; cmake ${MCM_HOME}/tests/system_tests/lp_scheduler/three_layer_conv_model
```

* Generate test bench and blobs
```
make generate_test_bench
```

* Check the blobs and test inputs
```
vamsikku@vamsikku-DESK:~/work/mcmCompiler/tests/system_tests/lp_scheduler/three_layer_conv_model/build$ ls *.blob
default_scheduler.blob  lp_scheduler.blob
vamsikku@vamsikku-DESK:~/work/mcmCompiler/tests/system_tests/lp_scheduler/three_layer_conv_model/build$ ls *.bin
input-0.bin  input-1.bin  input-2.bin  input-3.bin
vamsikku@vamsikku-DESK:~/work/mcmCompiler/tests/system_tests/lp_scheduler/three_layer_conv_model/build$ 
```

