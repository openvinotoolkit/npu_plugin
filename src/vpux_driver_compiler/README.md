# VPUX Driver Compiler
VPUX driver compiler package contains the driver compiler library, header of library, elf files and test tools.

## Folder Structure

```bash
── CiD_XXXX
   ├── CHANGES.txt
   ├── compilerTest.exe
   ├── data
   ├── lib
   ├── loaderTest.exe
   ├── pdb
   ├── profilingTest.exe
   ├── README.md
   ├── VPUXCompilerL0.h
   ├── vpuxCompilerL0Test.exe
   ├── vpuxCompilerL0Test.exe
   ├── vpux_driver_compiler.h
   └── vpux_elf
```
- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `vpu_elf` contains elf related files.
- `vpux_driver_compiler.h`  is the header file for exported functions.
- `compilerTest.exe`, `vpuxCompilerL0Test.exe`, `profilingTest.exe` and `loaderTest.exe`  are executables for test.

The Linux version does not include pdb folder.

## Running tests

### Basic tools
```bash
compilerTest googlenet-v1.xml googlenet-v1.bin output.net
compilerTest xxx.xml xxx.bin output.net config.file
profilingTest xxx.blob profiling-0.bin
loaderTest -v=1 (or -v=0)
```

### Unit test tools
vpuxCompilerL0Test is the test suit of the driver compiler:

To run vpuxCompilerL0Test, you need to export POR_PATH manually. E.g.
```
export POR_PATH=/path/to/om-vpu-models-por-ww46
```
You also need to export CID_TOOL manually to load the config for testing. E.g.
```
export CID_TOOL=/path/to/FLEX-CiD-Tools/release-tools
```
Currently, the test cases are defined in the config file under CID_TOOL path
```
vpuxCompilerL0Test
```

# Develop Info

### applications.ai.vpu-accelerators.vpux-plugin
The package is built from the repo

**Note: This package provides a thin wrapper/API of compiler to generate blob.**

The main entrance is `vclCompilerCreate`. Check full API demo - compilerTest.

- Basic work flow:

```C
...
vclCompilerCreate
...
vclCompilerGetProperties
...
/* If you want to query the supported layers of a network, please call following three lines. */
...
vclQueryNetworkCreate
...
/* vclQueryNetwork should be called twice, first time to retrieve data size, second time to get data. */
vclQueryNetwork
...
vclQueryNetworkDestroy
...
/* Fill buffer/weights with data read from command line arguments. Will set result blob size. */
...
vclExecutableCreate
...
vclExecutableGetSeriablizableBlob
...
blobSize > 0
blob = (uint8_t*)malloc(blobSize)
vclExecutableGetSeriablizableBlob
...
/* If log handle is created with vclCompilerCreate, can call vclLogHandleGetString to get last error message.*/
...
vclLogHandleGetString
...
logSize > 0
log = (char*)malloc(logSize)
vclLogHandleGetString
...
vclExecutableDestroy
vclCompilerDestroy
...

```
