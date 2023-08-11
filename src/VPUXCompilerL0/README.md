# Demo Usage
Notes about vpuxCompilerL0Test:
To run vpuxCompilerL0Test, you need to export POR_PATH manually. E.g.
```
export POR_PATH=/path/to/om-vpu-models-por-ww46
```
You also need to export CID_TOOL manually to load the config for testing. E.g.
```
export CID_TOOL=/path/to/FLEX-CiD-Tools/release-tools
```
Currently, vpuxCompilerL0Test supports the following model test cases:
1. mobilenet-v2 (smoke test)
2. googlenet-v1 (smoke test)
3. simple_function (smoke test, created by ngraph::Function)
4. resnet-50-pytorch
5. yolo_v4

## Linux

### Folder Structure

```bash
── CiD_Linux_XXXX
   ├── CHANGES.txt
   ├── compilerTest
   ├── lib
   ├── README.md
   ├── VPUXCompilerL0.h
   ├── vpuxCompilerL0Test
   ├── compilerTest
   ├── profilingTest
   └── loaderTest

```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `VPUXCompilerL0.h`  is the header file for exported functions.
- `compilerTest`, `vpuxCompilerL0Test`, `profilingTest` and `loaderTest` are executables for test.

```bash
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest googlenet-v1.xml googlenet-v1.bin output.net
LD_LIBRARY_PATH=./lib/ ./compilerTest xxx.xml xxx.bin output.net config.file
LD_LIBRARY_PATH=./lib/ ./profilingTest xxx.blob profiling-0.bin
LD_LIBRARY_PATH=./lib/ ./loaderTest -v=1 (or -v=0)
```
`output.net`  is the generated blob.

E.g if you want to run all the smoke tests in vpuxCompilerL0Test with the gtest_filter:
```
LD_LIBRARY_PATH=./lib/ ./vpuxCompilerL0Test --gtest_filter=*smoke*
```

## Windows

### Folder Structure

```bash
── CiD_WIN_XXXX
   ├── CHANGES.txt
   ├── compilerTest
   ├── lib
   ├── pdb
   ├── README.md
   ├── VPUXCompilerL0.h
   ├── vpuxCompilerL0Test.exe
   ├── compilerTest.exe
   └── profilingTest.exe
   └── loaderTest.exe
```
- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `VPUXCompilerL0.h`  is the header file for exported functions.
- `compilerTest.exe`, `vpuxCompilerL0Test.exe`, `profilingTest.exe` and `loaderTest.exe`  are executables for test.

### Windows (git bash)

```bash
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe googlenet-v1.xml googlenet-v1.bin output.net
PATH=$PATH:./lib/ ./compilerTest.exe xxx.xml xxx.bin output.net config.file
PATH=$PATH:./lib/ ./profilingTest.exe xxx.blob profiling-0.bin
PATH=$PATH:./lib/ ./loaderTest.exe -v=1 (or -v=0)
```

E.g if you want to run all the smoke vpuxCompilerL0Test tests with the gtest_filter:
```
PATH=$PATH:./lib/ ./vpuxCompilerL0Test --gtest_filter=*smoke*
```
### Windows (PowerShell)

```bash
cd .\CiD_WIN_XXXX\
$Env:Path +=";.\lib"
.\compilerTest.exe googlenet-v1.xml googlenet-v1.bin output.net
.\compilerTest.exe xxx.xml xxx.bin output.net config.file
.\profilingTest.exe xxx.blob profiling-0.bin
```
`output.net`  is the generated blob.

E.g if you want to run all the smoke vpuxCompilerL0Test tests with the gtest_filter:
```
.\vpuxCompilerL0Test --gtest_filter=*smoke*
```


# Develop Info

### applications.ai.vpu-accelerators.vpux-plugin
The lib is developed based on

**Note: This package provides a thin wrapper/API to generate blob.**

The main entrance is `vclCompilerCreate`. Check full API demo - compilerTest.

- Example:

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
vclLogHandleDestroy
...
vclExecutableDestroy
vclCompilerDestroy
...

```
