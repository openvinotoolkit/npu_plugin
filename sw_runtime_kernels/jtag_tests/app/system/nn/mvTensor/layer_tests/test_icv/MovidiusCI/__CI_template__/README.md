- The 1st step: build the testing application:
in `vpuip_2/validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/build` run `make -j8 all`
- The 2nd step: generate directories of test groups with test lists:
in `vpuip_2/validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/build` run `make MovidiusCI`
- The 3nd step: run test application for each group:  
in `vpuip_2/validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/MovidiusCI/SET<group index>` run `time make -j8 run`
