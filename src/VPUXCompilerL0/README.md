# Demo Usage
Notes about vpuxCompilerL0Test:
To run vpuxCompilerL0Test, you need to export POR_PATH manually. E.g.
```
export POR_PATH=/path/to/om-vpu-models-por-ww46
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
   └── profilingTest
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `VPUXCompilerL0.h`  is the header file for exported functions.
- `compilerTest`, `vpuxCompilerL0Test` and `profilingTest` are executables for test.

```bash
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest googlenet-v1.xml googlenet-v1.bin output.net
LD_LIBRARY_PATH=./lib/ ./compilerTest xxx.xml xxx.bin output.net config.file
LD_LIBRARY_PATH=./lib/ ./profilingTest xxx.blob profiling-0.bin
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
```
- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `VPUXCompilerL0.h`  is the header file for exported functions.
- `compilerTest.exe`, `vpuxCompilerL0Test.exe` and `profilingTest.exe` are executables for test.

### Windows (git bash)

```bash
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe googlenet-v1.xml googlenet-v1.bin output.net
PATH=$PATH:./lib/ ./compilerTest.exe xxx.xml xxx.bin output.net config.file
PATH=$PATH:./lib/ ./profilingTest.exe xxx.blob profiling-0.bin
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
vclExecutableDestroy
vclCompilerDestroy
...

```
