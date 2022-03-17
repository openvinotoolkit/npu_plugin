# Software kernels for VPUX

### Components
- [`kernels`](kernels) - source code of (SHAVE) cpp kernels  
- [`jtag_tests`](jtag_tests) - testing system for developing, optimization, debug and validation of kernels on board or symulator through JTAG  
- [firmware.vpu.iot](https://github.com/intel-innersource/firmware.vpu.iot) repository is necessary (should be available locally)  

to build/execute the tests and to compile the kernels for VPUX compiler.
[`firmware_vpu_revision.txt`](firmware_vpu_revision.txt) - file must contain:
- corresponding branch and/or commit hash of firmware.vpu.iot repo to work
- MV_TOOLS_VERSION which should be used to build act_shave code binaries in the following form
```
mv_tools_version:   <version, for example 21.11.3-internal>
```
### Build/execute JTAG tests
#### Prerequisites

Create `FIRMWARE_VPU_DIR` environment variable.
```
export FIRMWARE_VPU_DIR=<absolute path to firmware.vpu.iot repo>
```
firmware.vpu.iot repo should be checkouted on branch (or hash) pointed in `firmware_vpu_revision.txt`  
(submodules presilicon and schema should be updated)

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

### [Software layers agreements/description from VPUX NN compiler's point of view](https://docs.intel.com/documents/MovidiusInternal/vpu2/Common/SW/VPUX_NN_Compiler_SAS/VPUX_NN_Compiler_SAS.html#software-layers)

### Kernel's arrangements
ActShave sw kernel is represented in the VPUNN network (blob)
by fragments of precompiled elf file of the kernel (so called text segment and data segment).  
Text and data segments of the kernels are delivered together with VPUX network compiler
as separate binary files.  
The files are prepared by build procedure which can be done using [cmake script](kernels/CMakeLists.txt)

#### Compile/link the kernels to be added by VPUX compiler into the blob  
* See [Known issues](#known-issues)  

To prepare binaries the following steps should be done:
- create a temporary 'build' directory, cd into it
- call of cmake <path-to-kernels-dir> [options]
- call make

For detailed description of cmake options and corresponding examples, see [separate readme file](kernels/README.md)

The files are located in [`sw_runtime_kernels/kernels/prebuild`](kernels/prebuild) directory
and have names `sk.<entry point>.<platform>.<text or data as etension>.xdat`.
In each file the array filled by corresponding kernel segment and its size are defined in c/c++ syntax.
The array and size have the names
```
  unsigned char sk_<entry point>_<platform>_<text or data as etension>[] = { <hex values> };
  unsigned int sk_<entry point>_<platform>_<text or data as etension>_len = <len of array>;
```

#### Kernel creating/porting 
The main way of SW ActShave kernel execution is: one particular shave gets 
independent portion of input data located in NN CMX and returns the results 
into independent portion of output located in NN CMX.  
The kernel gets all input parameters as a one void pointer represented as uint32_t value  
example: [`void singleShaveSoftmax(uint32_t lParams)`](kernels/single_shave_softmax.cpp#L402)  
The content of the parameters pointed by [`lParams`](kernels/inc/param_softmax.h#L17) is a [contract between
vpux-compiler and particular sw kernel implementation](https://docs.intel.com/documents/MovidiusInternal/vpu2/Common/SW/VPUX_NN_Compiler_SAS/VPUX_NN_Compiler_SAS.html#software-layers)
The parameter structure can be any, but usually it includes:
- one or several input tensor descriptions,
- one or several output tensor descriptions,
- kernel specific parameters.  

In turn, tensor descriptions are represented as [MemRefData structure](kernels/inc/common_types.h#L78).  
Tensor data, dims and strides are pointed by dataAddr, dismAddr and stridesAddr correspondingly.
Kernel code can operate the `addr` fields as normal memory pointers,
which possibility is provided by windows-based virtual addressing feature of platform.  
Tensor dims and strides are written in 'memory' order
(dims[0] contains 'inner'dimension, dims[ndims - 1] contains 'outer' dimension).  
Strides are measured in bits.

To create SW kernel it is necessary:
- add kernel code in special source file in `sw_runtime_kernels` directory ([example](kernels/sigmoid_fp16.c))
  - the main kernel function (entry point) should be declared as `extern "C"`
  - if the kernel contains platform specific code
  the source file should be placed into platform specific  [subdirectory (3720)](kernels/3720) 
- add kernel parameters structure declaration in `sw_runtime_kernels/inc` directory ([example](kernels/inc/param_sigmoid.h))
which will be shared between vpux-compiler/vpux-plugin build and ActShave code moviCompile build
- Add script to prepare kernel binaries to be serialized into compiled network blob in `sw_runtime_kernels/kernels/prebuild`
([example and template](kernels/prebuild/singleShaveSoftmax.3010xx.sh)).
The files are prepared by special scripts in `sw_runtime_kernels/kernels/prebuild`
([example for softmax SW kernel](kernels/prebuild/singleShaveSoftmax.3010xx.sh))
- make the necessary preparations in vpux-compiler to provide
parsing, compilation and serialization through the compiler's dialects (["how to" reference](../src/vpux_compiler/docs/mtl_sw_layer_enabling.md))
- Add single layer vpux-compiler/vpux-plugin test (["how to"](https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/blob/5c6fd2be79a53cfde437f4d0d6c517a63a799f3d/src/vpux_compiler/docs/mtl_sw_layer_enabling.md), [lit mlir example](../tests/lit/mtl/act_shave/act_shave_gen_dma_sigmoid.mlir), [functional test example](https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/147))
- Add low-level JTAG tests as described in the next section. 

#### JTAG test creating/porting 
Low level sw-kernel JTAG testing system in vpux-plugin is being developed on the base of
[ICV tests in firmware.vpu.iot repo for UPA shave SW layers](https://github.com/intel-innersource/firmware.vpu.iot/tree/develop/validation/validationApps/system/nn/mvTensor).
Vpux-plugin JTAG testing system uses some elements from firmware.vpu.iot repo
including building system.
Low-level vpux-plugin JTAG tests builds and executes for two platforms KMB and MTL (on EVM KMB board and MTL moviSim correspondingly).
Kernels execute on ActShave processors (MTL) or on UPA shave processors (KMB).
The tests are based on CustomCPP test family.
The kernel test:
- is executed on VPU leon processor,
- is represented by the test class derived from [`CustomCppTests` template class](jtag_tests/app/layer_tests/test_icv/leon/tests/custom_cpp_tests.h#L30)
([example](jtag_tests/app/layer_tests/test_icv/leon/tests/custom_cpp_sigmoid.cpp#L22)),
- prepares an instance of parameter structure
declared for the kernel in `kernels/inc` directory,
- provides the pointer to kernel entry point function represented in the array prepared by
`xx` linux utility and included in the test c++ module, for example:  
```
...
#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.hswish_fp16.3010xx.text.xdat"
#else
...

```
- calculates the reference values and compare them with the values given by the tested kernel 
(or import precalculated values from a file),  
To create the test: 
- create special test source in `sw_runtime_kernels/jtag_tests/app/layer_tests/test_icv/leon/tests`,
for example, as copy of [sigmoid test](jtag_tests/app/layer_tests/test_icv/leon/tests/custom_cpp_sigmoid.cpp).
- Declare there the test class inside a unique test namespace ([example](jtag_tests/app/layer_tests/test_icv/leon/tests/custom_cpp_sigmoid.cpp#L16)).
- Override necessary class methods (initData, generateInputData, checkResults and others)
to prepare and provide kernel parameter structure and kernel entry point address,
generate inputs, reference outputs and compare results.
- Define in the kernel parameters structure header file 
the function to wrap kernel parameters structure into special common communication structure `BaseKernelParams`
([example](kernels/inc/param_sigmoid.h#L20)).
- Add entry point symbols into [svuSLKernels_EP.h](jtag_tests/app/nn/shave_lib/inc/layers/svuSLKernels_EP.h#L142) and
[jtag_tests/app/nn/shave_lib/shave/subdir.mk](jtag_tests/app/nn/shave_lib/shave/subdir.mk#L36) for KMB

[Another example of minimal necessary changes of simple sw kernel addition (hswish PR)](https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/48/files?authenticity_token=o5r2ig6Pe2TqS0WB1xIQhQ%2FkIHZc0MXvTZOfdwAAqH3wD3FZe1DKcl6v9%2BYlEySBaXLwdNTyyM9nKClz5MepGg%3D%3D&file-filters%5B%5D=.cpp&file-filters%5B%5D=.data&file-filters%5B%5D=.h&file-filters%5B%5D=.mk&file-filters%5B%5D=.text&file-filters%5B%5D=dotfile)  
( !!! PR does not contain the preparation and using of hexadecimal 
xdat ActShave kernel representation )

### Known issues
- [\[MOVICOMPILER\] Some code sequences are not compiled for 3720 act shave with O3](https://jira.devtools.intel.com/browse/EISW-26562) - 
in particular mvSubsoaces.cpp is not compiled for 3720 act shave with -O3

