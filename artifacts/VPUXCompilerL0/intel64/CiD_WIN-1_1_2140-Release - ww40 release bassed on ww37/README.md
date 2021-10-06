# Demo Usage

## Linux

### Folder Structure

```
── CiD_Linux_XXXX
   ├── data
   ├── lib
   ├── README.md
   ├── vpux_compiler_l0.h
   └── compilerTest
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest`  is an executable for test.

```
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest data/add_abc.xml data/add_abc.bin output.net
```

`output.net`  is the generated blob.

## Windows

### Folder Structure

```
── CiD_WIN_XXXX
   ├── data
   ├── lib
   ├── pdb
   ├── README.md
   ├── vpux_compiler_l0.h
   └── compilerTest.exe
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest.exe`  is an executable for test.

### Windows (git bash)

```
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe data/add_abc.xml data/add_abc.bin output.net
```
### Windows (PowerShell)

```
cd .\CiD_WIN_XXXX\
$Env:Path +=";.\lib"
.\compilerTest.exe .\data\add_abc.xml .\data\add_abc.bin output.net
```

`output.net`  is the generated blob.

# Develop Info

### kmb-plugin
The lib is developed based on

- Branch

```
releases/2021/4_vpux_ww37
```

- Commit hash

```
4e8b21ba2e15da1d8c67d8118b2622707a8b2133
```
**Note: We have modifications on kmb-plugin and provide a thin wrapper/API to generate blob.**

The main entrance is `getVPUXCompilerL0`. Check full API demo [here](https://gitlab.devtools.intel.com/flex-plaidml-team/kmb-plugin/-/blob/VPUXCompilerL0/tests/umd/test/compilerTest.c). If you can't access the link, please contact Wang, Xin1(xin1.wang@intel.com) to request access.

- Definition:
```
gc_result_t getSerializableBlob(vcl_executable_handle_t exe, uint8_t* blob, uint32_t blobSize)
```
- Example:
```
...
getVPUXCompilerL0 = (gc_result_t(*)(vpux_compiler_l0_t*))LIBFUNC(handle, GET_VPUX_COMPILER_L0);
vpux_compiler_l0_t vcl;
getVPUXCompilerL0(&vcl);
vcl.methods.createCompiler(...)
...
uint8_t* modelIR;
/* Memory layout
numOfElements;  // Type: uint32_t
bufferSize;     // Type: uint32_t
bufferData;     // Memory block
weights;        // Type: uint32_t
weightsData;    // Memory block
*/
...
uint8_t* blob;
uint32_t blobSize = 0;
...
/* Fill buffer/weights with data read from command line arguments. Will set result blob size. */
...
ret = vcl.methods.generateSerializableBlob(compilerHandle, exeDesc, modelIR, &blobSize, exeHandle);
...
blobSize > 0
blob = (uint8_t*)malloc(blobSize)
ret = vcl.methods.getSerializableBlob(exeHandle, blob, blobSize);
...
vcl.methods.destoryExecutable(...)
vcl.methods.destroyCompiler(...)
...

```


### OpenVINO

- Branch

```
releases/2021/4
```

- Commit hash

```
f00dc87a92cca4db8177d480e14d2558dd989bc5
```
