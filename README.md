# mcmCompiler
Movidius C++ Computational Model Compiler

## Setup
Environment variable `MCM_HOME` must be set and pointing to the root directory of the project.

For example:

```
export MCM_HOME=~/mcmCompiler 
```

## Building
```
cd mcmCompiler
mkdir build && cd build && cmake ..
make -j8
```

## Running
```
cd mcmCompiler
cd build/examples
./cm_resnet18
```
