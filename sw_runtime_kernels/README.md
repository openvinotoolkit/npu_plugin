# Software kernels for VPUX

### Components
- `kernels` - source code of (SHAVE) cpp kernels  
- `jtag_tests` - testing system for developing, optimization, debug and validation of kernels on board or symulator through JTAG  
- vpuip_2 repository is necessary (should be available locally)  

to build/execute the tests and to compile the kernels for VPUX compiler.
`vpuip_2_revision.txt` file must contain corresponding branch and/or commit hash of vpuip_2 repo to work

### Compile/link the kernels to be added by VPUX compiler into the blob  
### Build/execute JTAG tests  
#### Prerequisites  
Create `VPUIP_2_Directory` environment variable.  
```
export VPUIP_2_Directory=<absolute path to vpuip_2>
```
vpuip_2 repo should be checkouted on branch (or hash) pointed in `vpuip_2_revision.txt`  
(all the vpuip_2 submodules/LFS should be updated/fetched)

#### Build/execute the tests
build/execute for MeteorLake:  
in `sw_runtime_kernels/jtag_tests/app/layer_tests/test_icv/build` run:  
`make -j8 all CONFIG_FILE=.config_sim_3720xx_release` to build  
`make start_simulator CONFIG_FILE=.config_sim_3720xx_release srvPort=30002 &` to start MTL debug simulator  
`make CONFIG_FILE=.config_sim_3720xx_release run srvIP=127.0.0.1 srvPort=30002 CONFIG_TEST_FILTER="*" CONFIG_TEST_MODE_FULL=y` to run tests  


build/execute for KeemBay:  
in `sw_runtime_kernels/jtag_tests/app/layer_tests/test_icv/build` run:  
`make -j8 all CONFIG_FILE=.config` to build  
`make start_server CONFIG_FILE=.config &` to start jtag moviDebugServer  
`make -j8 CONFIG_FILE=.config run CONFIG_TEST_FILTER="*" CONFIG_TEST_MODE_FULL=y` to run tests  

