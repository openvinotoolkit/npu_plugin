# How to build kernel binaries

## Prerequisites

- [firmware.vpu.iot](https://github.com/intel-innersource/firmware.vpu.iot) repository is required

## Environment variables

- [FIRMWARE_VPU_DIR](FIRMWARE_VPU_DIR)
  absolute path to firmware.vpu.iot workspace (required)
- [MV_TOOLS_DIR](MV_TOOLS_DIR)
  absolute path to tools directory (required)
- [MV_TOOLS_VERSION](MV_TOOLS_VERSION)
  tools version string (optional). If not set, tools version is determined by the [firmware_vpu_revision.txt](firmware_vpu_revision.txt) file

## Command line

Two usual stages: cmake & then make.

```
mkdir build ; cd build

cmake src-dir-path [options]

make
```

where [src-dir-path](src-dir-path) is an absolute path to [sw_runtime_kernels/kernels](sw_runtime_kernels/kernels) directory, and options is -Doption list.
option must be:

- [BUILD_BLOB_BINARIES=ON|OFF](BUILD_BLOB_BINARIES=ON|OFF)
  specifies whether to build 'blob' binary files (.text & .data)
- [BUILD_JTAG_BINARIES-ON|OFF](BUILD_JTAG_BINARIES-ON|OFF)
  specifies whether to build 'jtag tests' binary files (.xdata)
- [BUILD_STD_KERNELS=ON|OFF](BUILD_STD_KERNELS=ON|OFF)
  specifies whether to build the mostly kernels, including Management Kernel (nnActEntry) (description: descrip/*.txt)
- [BUILD_PSS_KERNELS=ON|OFF](BUILD_PSS_KERNELS=ON|OFF)
  specifies whether to build PSS tests - targeted kernels (description: descrip/pss/*.txt)
- [CUSTOM_KERNELS_DIR=descrip-dir-path](CUSTOM_KERNELS_DIR=descrip-dir-path)
  specifies path to an alternative directory containing kernel description file(s). Default is [sw_runtime_kernels/kernels/descrip/](sw_runtime_kernels/kernels/descrip/)
- [CUSTOM_KERNELS_LIST=kernels-list](CUSTOM_KERNELS_LIST=kernels-list)
  specifies a semicolon-separated list of description file names (possibly in quotes), for kernels to build. Default: all files found in description directory

## Installation of binaries

```
make install
```

## Kernel description files (descrip/*.txt)

Description file is the text file, which will be include()d into cmake script and customize build settings according to particular kernel.
Usually but not necessarily description file contains one or more set() or list() cmake statements which assign a values to a dedicated variables.
As for cmake scripts, '#' character marks a comment line.

CAUTION: since description file include()d into cmake script, it can't be isolated from cmake script context.
So, you can unintentionally change the behaviour of cmake and even make it wrong. Be careful!

Variables intended for use/set in description files:

- [kernel_entry](kernel_entry)
  a string which specifies kernel entry point name (optional). Default: "${kernel_src}" without path and extension(s) (suffices).
- [kernel_src](kernel_src)
  a string which specifies kernel source file name, without path and relative to "${kernel_src_dir}". Required.
- [kernel_src_dir](kernel_src_dir)
  a string which specifies a path to directory containing "${kernel_src}" file.
  Can be absolute, or relative to [sw_runtime_kernels/kernels](sw_runtime_kernels/kernels). Optional; default is "src".
- [kernel_cpunum](kernel_cpunum)
  a string which specifies a target chip level. Optional; default is "3010".
  Sources compiled with "-mcpu=${kernel_cpunum}xx" option.
  Built binaries' names have suffix in the form of ".${kernel_cpunum}xx" .
- [optimization_opts](optimization_opts)
  a string which specifies an optimization option for source files compilation. Optional; default is "-O3" .
- [include_dirs_list](include_dirs_list)
  a string which specifies an additional C/C++ include directories list in the cmake form ("dir1;dir2;etc").
  List is parsed and each directory is prepended by "-I" automatically. Optional.
- [define_symbols_list](define_symbols_list]
  a string which specifies an additional C/C++ #define symbols list in the cmake form ("sym1;sym2;etc").
  List is parsed and each symbol is prepended by "-D" automatically. Optional.
- [always_inline](always_inline)
  a string which specifies whether compilation use inlined code (-DCONFIG_ALWAYS_INLINE). Optional; default is "yes".
  Can also be checked in description code in if() statement(s) to change compile/link behaviour (e.g. add extra source files).
- [extra_src_list](extra_src_list)
  a string which specifies an additional C/C++ source files (absolute paths) which will be compiled and linked to "${kernel_src}" file.
  List must be in the cmake form ("src1;src2;etc"), it is parsed automatically.

Examples:

dummy.txt

```
set(kernel_src "dummy.cpp")
```

singleShaveSoftmax.txt

```
set(kernel_src "singleShaveSoftmax.cpp")

set(optimization_opts "") # -O3
set(always_inline "yes") # "no"

if(NOT always_inline STREQUAL "yes")
  set(extra_src_list "${CMAKE_SOURCE_DIR}/common/src/mvSubspaces.cpp")
endif()
```

nnActEntry.txt

```
set(kernel_src "nnActEntry.cpp")
set(kernel_src_dir "act_runtime/src")

set(include_dirs_list
  "${FIRMWARE_VPU_DIR}/drivers/errors/errorCodes/inc"
  "${FIRMWARE_VPU_DIR}/drivers/hardware/registerMap/inc"
  "${FIRMWARE_VPU_DIR}/drivers/nn/inc"
  "${FIRMWARE_VPU_DIR}/drivers/resource/barrier/inc"
  "${FIRMWARE_VPU_DIR}/drivers/shave/svuCtrl_3600/inc"
  "${FIRMWARE_VPU_DIR}/drivers/shave/svuL1c/inc"
  "${FIRMWARE_VPU_DIR}/drivers/shave/svuShared_3600/inc"
  "${FIRMWARE_VPU_DIR}/drivers/vcpr/perf_timer/inc"
  "${FIRMWARE_VPU_DIR}/system/nn_mtl/act_runtime/inc"
  "${FIRMWARE_VPU_DIR}/system/shave/svuCtrl_3600/inc"
  "${CMAKE_SOURCE_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/inc"
  "${CMAKE_SOURCE_DIR}/../jtag_tests/app/nn/common/inc"
)

set(extra_src_list
  "${CMAKE_SOURCE_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/src/nn_fifo_manager.cpp"
)

# performance measurement support

list(APPEND define_symbols_list
  "LOW_LEVEL_TESTS_PERF"
)

list(APPEND extra_src_list
  "${CMAKE_SOURCE_DIR}/../jtag_tests/app/act_shave_lib/leon/common_runtime/src/nn_perf_manager.cpp"
)
```
