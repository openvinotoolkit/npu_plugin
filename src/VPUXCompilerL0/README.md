# Demo Usage

## Linux

### Folder Structure

```
── CiD_Linux_XXXX
   ├── data
   ├── lib
   ├── README.md
   ├── vpux_compiler_l0.h
   ├── compilerTest
   ├── compilerThreadTest
   └── compilerThreadTest2
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest` `compilerThreadTest` `compilerThreadTest2` are executables for test.

```
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest data/add_abc.xml data/add_abc.bin output.net
LD_LIBRARY_PATH=./lib/ ./compilerTest data/add_abc.xml data/add_abc.bin output.net FP16 C FP16 C ./data/config.file
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest data/add_abc.xml data/add_abc.bin
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest2 data/add_abc.xml data/add_abc.bin
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
   ├── compilerTest
   ├── compilerThreadTest
   └── compilerThreadTest2
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest` `compilerThreadTest` `compilerThreadTest2` are executables for test.

### Windows (git bash)

```
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe data/add_abc.xml data/add_abc.bin output.net
PATH=$PATH:./lib/ ./compilerTest.exe data/add_abc.xml data/add_abc.bin output.net FP16 C FP16 C ./data/config.file
PATH=$PATH:./lib/ ./compilerThreadTest data/add_abc.xml data/add_abc.bin
PATH=$PATH:./lib/ ./compilerThreadTest2 data/add_abc.xml data/add_abc.bin
```
### Windows (PowerShell)

```
cd .\CiD_WIN_XXXX\
$Env:Path +=";.\lib"
.\compilerTest.exe .\data\add_abc.xml .\data\add_abc.bin output.net
.\compilerTest.exe .\data\add_abc.xml .\data\add_abc.bin output.net FP16 C FP16 C ./data/config.file
.\compilerThreadTest .\data\add_abc.xml .\data\add_abc.bin
.\compilerThreadTest2 .\data\add_abc.xml .\data\add_abc.bin
```

`output.net`  is the generated blob.

# Develop Info

### kmb-plugin
The lib is developed based on

- Branch

```
master
```

- Commit hash

```
11bad15dce79b395c88c74515bdf0bb8ae9cb3cd
```
**Note: We have modifications on kmb-plugin and provide a thin wrapper/API to generate blob.**

The main entrance is `vclCompilerCreate`. Check full API demo - compilerTest | compilerThreadTest | compilerThreadTest2.

- Example:
```
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

- Commit hash

```
ce51b62b7008c75a2871b5b8e3eef00fa9972ee6
```
