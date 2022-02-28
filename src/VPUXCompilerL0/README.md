# Demo Usage

## Linux

### Folder Structure

```bash
── CiD_Linux_XXXX
   ├── data
   ├── lib
   ├── README.md
   ├── vpux_compiler_l0.h
   ├── compilerTest
   ├── compilerThreadTest
   ├── compilerThreadTest2
   └── profilingTest
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest` `compilerThreadTest` `compilerThreadTest2` `profilingTest` are executables for test.

```bash
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest googlenet-v1.xml googlenet-v1.bin output.net
LD_LIBRARY_PATH=./lib/ ./compilerTest xxx.xml xxx.bin output.net config.file
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest googlenet-v1.xml googlenet-v1.bin
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest2 googlenet-v1.xml googlenet-v1.bin
LD_LIBRARY_PATH=./lib/ ./profilingTest xxx.blob profiling-0.bin
```

`output.net`  is the generated blob.

## Windows

### Folder Structure

```bash
── CiD_WIN_XXXX
   ├── data
   ├── lib
   ├── pdb
   ├── README.md
   ├── vpux_compiler_l0.h
   ├── compilerTest.exe
   ├── compilerThreadTest.exe
   ├── compilerThreadTest2.exe
   └── profilingTest.exe
```

- `data` contains an xml and bin for test. E.g. if you want to test resnet-50, you can get its IR from `$KMB_PLUGIN_HOME/temp/models/src/models/KMB_models/FP16/resnet_50_pytorch/`
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest.exe` `compilerThreadTest.exe` `compilerThreadTest2.exe` `profilingTest.exe` are executables for test.

### Windows (git bash)

```bash
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe googlenet-v1.xml googlenet-v1.bin output.net
PATH=$PATH:./lib/ ./compilerTest.exe xxx.xml xxx.bin output.net config.file
PATH=$PATH:./lib/ ./compilerThreadTest googlenet-v1.xml googlenet-v1.bin
PATH=$PATH:./lib/ ./compilerThreadTest2 googlenet-v1.xml ad_abc.bin
PATH=$PATH:./lib/ ./profilingTest.exe xxx.blob profiling-0.bin
```

### Windows (PowerShell)

```bash
cd .\CiD_WIN_XXXX\
$Env:Path +=";.\lib"
.\compilerTest.exe googlenet-v1.xml googlenet-v1.bin output.net
.\compilerTest.exe xxx.xml xxx.bin output.net config.file
.\compilerThreadTest googlenet-v1.xml googlenet-v1.bin
.\compilerThreadTest2 googlenet-v1.xml googlenet-v1.bin
.\profilingTest.exe xxx.blob profiling-0.bin
```

`output.net`  is the generated blob.

# Develop Info

### applications.ai.vpu-accelerators.vpux-plugin
The lib is developed based on

- Branch

```
master
```

**Note: This package provides a thin wrapper/API to generate blob.**

The main entrance is `vclCompilerCreate`. Check full API demo - compilerTest | compilerThreadTest | compilerThreadTest2.

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

### OpenVINO

- Branch

```
master
```
