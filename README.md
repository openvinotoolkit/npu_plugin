# mcmCompiler
Movidius C++ Computational Model Compiler

## Setup
Environment variable `MCM_HOME` must be set and pointing to the root directory of the project.
Environment variable `MDK_HOME` must be set and pointing to the root directory of the mdk project.

For example:

```
export MCM_HOME=~/mcmCompiler 
export MDK_HOME=~/mdk 
```

You must have Metis C++ library and header installed. Recommendation is to install Metis using apt-get.

sudo apt-get install metis

Note: apt-get brings binaries. Metis can be built from source but GKLIB in metis uses GKRAND from C Compiler which can differ from OS to OS. 
FYI..Metis can be found at the following link:
http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
To build METIS, follow the instructions in the file metis-5.1.0/Install.txt. 

#### Python CompositionAPI requirements
- numpy 1.16.4
- tensorflow 1.13.1

## Building
```
git clone --recursive https://github.com/movidius/mcmCompiler.git
cd mcmCompiler
git submodule update --init
mkdir build && cd build && cmake ..
make -j8
```

## Running
```
cd mcmCompiler
cd build/examples
./cm_resnet50
```
## Troubleshooting

#### Thrown `terminate called after throwing an instance of 'std::logic_error' what():  basic_string::_M_construct null not valid` during the execution of compilation

Missing `MCM_HOME` environment variable, it must be always set.

#### Thrown `OverflowError: in method 'conv2D', argument 6 of type 'unsigned short'` in the Python CompositionAPI bridge

Invalid numpy version, must be 1.16.4
