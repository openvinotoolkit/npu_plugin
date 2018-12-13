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

## Building
```
cd mcmCompiler
git submodule update --init
cd KeemBayFBSchema
python3 import_schema.py
mkdir compiledSchema
make
cd ..
mkdir build && cd build && cmake ..
make -j8
```
## Apply Patch
```
# 24-Aug-2018: patch to revert certain updates to  movidius/mdk repository.

Purpose: MCMCompiler does not support certain recent updates to the movidius MDK. Apply the following patch to revert certain commits from your local mdk branch. You may see warning messages about trailing whitespace that may be ignored.

>cd $MDK_HOME
>cp $MCM_HOME/patch/remove_eltwise_updates.patch .
>git apply remove_eltwise_updates.patch
```

## Running
```
cd mcmCompiler
cd build/examples
./cm_resnet50
```
