# How to build kernel binaries

## Prerequisites

- [firmware.vpu.iot](https://github.com/intel-innersource/firmware.vpu.iot) repository is required

## Environment variables

- FIRMWARE_VPU_DIR: absolute path to firmware.vpu.iot workspace (required)
- MV_TOOLS_DIR: absolute path to tools directory (required)
- MV_TOOLS_VERSION: tools version string (optional). If not set, tools version is determined by the firmware_vpu_revision.txt file

## Command line

Two usual stages: cmake & then make.

```
mkdir build ; cd build
cmake <src-dir-path> [options]
```

where src-dir-path is an absolute path to path to sw_runtime_kernels/kernels directory, and options is -D<option> list.
option must be:

- BUILD_BLOB_BINARIES=ON|OFF
  specifies whether to build 'blob' binary files (.text & .data)
- BUILD_JTAG_BINARIES-ON|OFF
  specifies whether to build 'jtag tests' binary files (.xdata)
- BUILD_STD_KERNELS=ON|OFF
  specifies whether to build the mostly kernels, including Management Kernel (nnActEntry) (description: descrip/*.txt)
- BUILD_PSS_KERNELS=ON|Off
  specifies whether to build PSS tests - targeted kernels (description: descrip/pss/*.txt)
- CUSTOM_KERNELS_DIR=<descrip-dir-path>
  specifies path to an alternative directory containing kernel description file(s). Default is sw_runtime_kernels/kernels/descrip/
- CUSTOM_KERNELS_LIST=<kernels-list>
  specifies a semicolon-separated list of description file names, for kernels to build. Default: all files found in description directory

## Binaries installation

## Kernel description files (descrip/*.txt)

