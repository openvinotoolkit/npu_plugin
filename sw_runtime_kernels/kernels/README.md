# How to build kernel binaries

## Environment variables

- `FIRMWARE_VPU_DIR` - absolute path to firmware.vpu.client workspace
- `MV_TOOLS_DIR` - absolute path to tools directory
- `MV_TOOLS_VERSION` - tools version string

## CMake options used in vpux-plugin build

- `ENABLE_SHAVE_BINARIES_BUILD=AUTO|ON|OFF`
    - `ON` - will try always to build sw layers and report error if `MV_TOOLS_DIR` or `MV_TOOLS_VERSION` were not found
    - `OFF` (default) - will ignore sw layers build, prebuilt binaries will be used
- `ENABLE_MANAGEMENT_KERNEL_BUILD=ON|OFF` - management kernel is treated separately because it also requires dependencies from `vpu.firmware` repository
    - `OFF` (default) - prebuilt binary for management kernel will be used
    - `ON` - will try to build management kernel and report an error if `MV_TOOLS_DIR`, `MV_TOOLS_VERSION` or `FIRMWARE_VPU_DIR` were not found
- `ENABLE_PSS_BINARIES_BUILD=ON|OFF`
    - `ON` (default) - build pss kernels if tools were found
    - `OFF` - ignore pss kernels build

## Kernel description files (descrip/*.txt)

Description file is the text file, which will be included into cmake script by include() statement and customize build options for a particular kernel.
Usually but not necessarily description file contains one or more set() or list() cmake statements which assign a values to a dedicated variables.
As for cmake scripts, '#' character marks a comment line.

CAUTION: since description file is included into cmake script, it can't be isolated from cmake script context.
So, you can unintentionally change the behaviour of cmake and even make it wrong. Be careful!

Variables intended to be used/set in description files:

- `kernel_entry`
  a string which specifies kernel entry point name. Optional; default is `"${kernel_src}"` without path and filename suffixes (extensions).
- `kernel_src`
  a string which specifies kernel source file name, without path and relative to `"${kernel_src_dir}"` . Required.
- `kernel_src_dir`
  a string which specifies a path to directory containing `"${kernel_src}"` file.
  Can be absolute, or relative to [sw_runtime_kernels/kernels](.). Optional; default is `"src"` .
- `kernel_cpunum`
  a string which specifies a target chip level. Optional; default is `"3720"`.
  Sources are compiled with `"-mcpu=${kernel_cpunum}xx"` option.
  Built binaries' names have suffix in the form of `".${kernel_cpunum}xx"` .
- `optimization_opts`
  a string which specifies an optimization option for compilation of source files. Optional; default is `"-O3"` .
- `include_dirs_list`
  a string which specifies an additional C/C++ include directories list in the cmake form (`"dir1;dir2;etc"`).
  The list is parsed and each directory is prepended by `"-I"` prefix automatically. Optional.
- `define_symbols_list`
  a string which specifies an additional C/C++ #define symbols list in the cmake form (`"sym1;sym2;etc"`).
  The list is parsed and each symbol is prepended by `"-D"` prefix automatically. Optional.
- `always_inline`
  a string which specifies whether compilation use inlined code (`-DCONFIG_ALWAYS_INLINE`) or not. Optional; default is `"no"` .
  Can also be checked in description code in if() statement(s) to change compile/link behaviour (e.g. add extra source files).
- `extra_src_list`
  a string which specifies an additional C/C++ source files (absolute paths) which will be compiled and linked together with `"${kernel_src}"` file to form the output binary.
  The list must be in the cmake form (`"src1;src2;etc"`), it is parsed automatically.
- `link_script_file`
  a string which specifies custom link 'ldscript' file. Optional; default is `"${CMAKE_SOURCE_DIR}/prebuild/shave_kernel.ld"` .
  For existing kernels only ManagementKernel (nnActEntry) uses different ldscript; for the rest of kernels default ldscript should be enough.
- `kernel_descrip_path`
  a string which specifies an absolute path to description file directory; can be used to include another (e.g. 'common') description file.
  Prepared by cmake script automatically; description file can use it.
- `binary_subdir`
  a string which specifies a particular subdirectory relative to `"${target-binary-dir}"` directory (i.e. `"${target-binary-dir}/${binary_subdir}"`). See PSS description files as an examples.

Examples:

### dummy.txt

```
set(kernel_src "dummy.cpp")

set(always_inline "yes")
```

### singleShaveSoftmax.txt

```
set(kernel_src "singleShaveSoftmax.cpp")

set(always_inline "yes")

if(NOT always_inline STREQUAL "yes")
  set(extra_src_list "${CMAKE_SOURCE_DIR}/common/src/mvSubspaces.cpp")
endif()
```
