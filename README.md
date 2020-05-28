# OpenVINO KMB plugin

## Prerequsites

### Yocto SDK

To build ARM64 code for KMB board the Yocto SDK is required. It can be installed with the following commands:

```bash
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/development/20200420-2100/oecore-x86_64-aarch64-toolchain-1.0.sh && \
    chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
    ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
    rm oecore-x86_64-aarch64-toolchain-1.0.sh
```

### Git projects

The following projects are used and must be cloned including git submodules update:

* [DLDT Project]
* [KMB Plugin Project]

### Environment variables

The testing command assumes that the KMB board was setup and is avaialble via ssh.

The following environment variables should be set:

* The `DLDT_HOME` environment variable to the [DLDT Project] cloned directory.
* The `KMB_PLUGIN_HOME` environment variable to the [KMB Plugin Project] cloned directory.
* The `KMB_BOARD_HOST` environment variable to the hostname or ip addess of the KMB board.
* The `KMB_WORK_DIR` environment variable to the working directory on the KMB board.

## Manual build

### Dependencies

1. To figure out dependencies for dldt checkout the script:

    ```https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/build-after-clone.sh```

2. In case of CMake version is too low error use ```sudo snap install --classic cmake``` and binary will be ```/snap/bin/cmake```

    For building ARM64 using cmake from the Yocto SDK is recomended

3. To initialize all submodlues (or reinitialize when branch changes) you have to execute `git submodule update --init --recursive` inside a repo dir

4. Also it could be required to manually initialize `git lfs pull` especially if you have error message:

    `lib**.so: file format not recognized; treating as linker script`

### Build for X86_64

The X86_64 build is needed to get reference results for the tests.

1. Move to [DLDT Project] base directory and build it with the following commands:

    ```bash
    mkdir -p $DLDT_HOME/build-x86_64
    cd $DLDT_HOME/build-x86_64
    cmake \
        -D ENABLE_MKL_DNN=ON \
        -D ENABLE_TESTS=ON \
        -D ENABLE_BEH_TESTS=ON \
        -D ENABLE_FUNCTIONAL_TESTS=ON \
        ..
    make -j8
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build-x86_64
    cd $KMB_PLUGIN_HOME/build-x86_64
    cmake \
        -D InferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build-x86_64 \
        ..
    make -j8
    ```

### Build for ARM64

1. Move to [DLDT Project] base directory and build it with the following commands:

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $DLDT_HOME/build-aarch64 ; \
        cd $DLDT_HOME/build-aarch64 ; \
        cmake \
            -D ENABLE_MKL_DNN=ON \
            -D ENABLE_TESTS=ON \
            -D ENABLE_BEH_TESTS=ON \
            -D ENABLE_FUNCTIONAL_TESTS=ON \
            -D THIRDPARTY_SERVER_PATH="http://nnt-srv01.inn.intel.com/dl_score_engine/" \
            .. ; \
        make -j8 ; \
    )
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $KMB_PLUGIN_HOME/build-aarch64 ; \
        cd $KMB_PLUGIN_HOME/build-aarch64 ; \
        cmake \
            -D InferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build-aarch64 \
            .. ; \
        make -j8 ; \
    )
    ```

### Manual MCM Compiler build 

#### Known issues 
##### libcm.so is not downloaded via git lfs  

You may encounter the following problem: 
```
/usr/bin/ld:../../../artifacts/mcmCompilerInstallPackage/lib/libcm.so: file format not recognized; treating as linker script
/usr/bin/ld:../../../artifacts/mcmCompilerInstallPackage/lib/libcm.so:1: syntax error
collect2: error: ld returned 1 exit status
```
And inside libcm.so you see text like this:  
```
version https://git-lfs.github.com/spec/v1
oid sha256:1246a6fdd15e23f809add06c7c2f76089d177cd4ac386e144778e2d1eadde9d7
size 6094808
```
This, possibly, mean, that you have lfs filter which is not allow you to download *.so files. Please take a look inside /home/$USER/.gitconfig file, and remove or extend filter in it: 
```sh
gedit /home/$USER/.gitconfig 
```
Filter, which not allow to download .so files
```
[lfs]
	fetchinclude = *.jpg,*.png,*.gif,*.bmp
```

#### How to build mcmCompiler for kmb-plugin
1. Clone the repository:
```sh
git clone git@github.com:movidius/mcmCompiler.git
```
2. Run the following script:
```sh
cd mcmCompiler && \
git checkout <The branch you need> && \
git submodule update --init --recursive && \
mkdir build && mkdir install && export MCM_HOME=$(pwd) && git rev-parse HEAD > install/revision.txt && cd build && \
cmake -DCMAKE_INSTALL_PREFIX=$MCM_HOME/install -DCMAKE_BUILD_TYPE=Release \
-DMCM_COMPILER_BUILD_PYTHON=OFF -DMCM_COMPILER_BUILD_TESTS=OFF \
-DMCM_COMPILER_BUILD_EXAMPLES=OFF -DMCM_COMPILER_FORCE_BUILD_LEMON=ON .. && make -j8 && make install

```
* The built package is located in the "$MCM_HOME/install" folder.
* The current revision of mcmCompiler is stored in the revision.txt file.

#### How to build kmb-plugin using custom mcmCompiler

* Currently mcmCompiler is a pre-built package.
* Default path: $KMB_PLUGIN_HOME/artifacts/mcmCompilerInstallPackage.
* To use a specific package, you do not need to delete the existing default package in kmb-plugin storage.

```
export KMB_PLUGIN_HOME=<path to kmb-plagin> && \
export DLDT_HOME=<path to dldt> && \
export MCM_HOME=<path to mcmCompiler> && \
mkdir -p $KMB_PLUGIN_HOME/build-x86_64 && \
cd $KMB_PLUGIN_HOME/build-x86_64 && \
cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build-x86_64 -DmcmCompiler_DIR=$MCM_HOME/install/share/cmake/mcmCompiler/ .. && make -j8
```

#### How to integrate mcmCompiler to kmb-plugin

```
export KMB_PLUGIN_HOME=<path to kmb-plagin> && \
export MCM_HOME=<path to mcmCompiler> && \
rm -rf $KMB_PLUGIN_HOME/artifacts/mcmCompilerInstallPackage/* && \
cp -r $MCM_HOME/install/* $KMB_PLUGIN_HOME/artifacts/mcmCompilerInstallPackage/ && \
git checkout -b <name_of_new_branch> && git add -A && git commit -m "integrate new version mcmCompiler" 
```

### Manual vpualHost build

#### How to build vpualHost for kmb-plugin
1. To build ARM64 code you need [Yocto SDK].
2. Clone the repository:
```sh
git clone git@github.com:movidius/vpualHost.git
```
3. Run the following script:
```sh
source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
cd vpualHost && git checkout <The branch you need> && git submodule update --init --recursive && \
mkdir build_aarch && mkdir install && export VPUAL_HOME=$(pwd) && git rev-parse HEAD > install/revision.txt && cd build_aarch && \
cmake -DCMAKE_INSTALL_PREFIX=$VPUAL_HOME/install -DCMAKE_BUILD_TYPE=Release .. && make -j8 && make install
```
* The built package is located in the "$VPUAL_HOME/install" folder.
* The current revision of vpualHost is stored in the revision.txt file.

#### How to build kmb-plugin using custom vpualHost

* Currently vpualHost is a pre-built package.
* Default path: $KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage.
* To use a specific package, you do not need to delete the existing default package in kmb-plugin storage.

```
export KMB_PLUGIN_HOME=<path to kmb-plugin> && \
export DLDT_HOME=<path to dldt> && \
export VPUAL_HOME=<path to vpualHost> && \
mkdir -p $KMB_PLUGIN_HOME/build_aarch && \
cd $KMB_PLUGIN_HOME/build_aarch && \
cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build_aarch -DvpualHost_DIR=$VPUAL_HOME/install/share/vpualHost/ .. && make -j8
```

#### How to integrate vpualHost to kmb-plugin

```
export KMB_PLUGIN_HOME=<path to kmb-plugin> && \
export VPUAL_HOME=<path to vpualHost> && \
rm -rf $KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage/* && \
cp -r $VPUAL_HOME/install/* $KMB_PLUGIN_HOME/artifacts/vpualHostInstallPackage/ && \
git checkout -b <name_of_new_branch> && git add -A && git commit -m "integrate new version vpualHost"
```

## Deployment to KMB board

Deploy OpenVINO artifacts to the KMB board:

```bash
rsync -avz --exclude "*.a" $DLDT_HOME/bin/aarch64/Release root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

Deploy OpenVINO dependencies to the KMB board (replace `<ver>` with actual latest version which were downloaded by OpenVINO CMake script):

```bash
rsync -avz $DLDT_HOME/inference-engine/temp/tbb_yocto/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $DLDT_HOME/inference-engine/temp/openblas_<ver>_yocto_kmb/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $DLDT_HOME/inference-engine/temp/opencv_<ver>_yocto_kmb/opencv/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
```

Mount the HOST `$DLDT_HOME/inference-engine/temp` directory to the KMB board as a remote SSH folder.
Run the following commands on the KMB board for this:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
mkdir -p $KMB_WORK_DIR/temp
sshfs <username>@<host>:$DLDT_HOME/inference-engine/temp $KMB_WORK_DIR/temp
```

**Note:** to unmount the HOST `$DLDT_HOME/inference-engine/temp` directory from the KMB board use the following command:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
fusermount -u $KMB_WORK_DIR/temp
```


## Testing on KMB board

### Run Layer Tests

These tests could be run on HOST or KMB-board. To be able to run test on KMB-board side you need to provide `IE_KMB_TESTS_DUMP_PATH` so the test framework can found compiled networks for tests. Please see how to do it in secsion [Target networks regression tests](#target-networks-regression-tests). But in any case (HOST or KMB-board) command line text is the same.

* Run the following command to launch Layer Tests:

```bash
$DLDT_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*LayerTests*
```

* If you want to run all Layer Tests including disabled ones then run this command:

```bash
$DLDT_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*LayerTests* --gtest_also_run_disabled_tests
```


### Target networks regression tests

#### Select testing plugin

The `IE_KMB_TESTS_DEVICE_NAME` environement should be set both on HOST and KMB Board to the desired target plugin for tests:

* `KMB` to use KMB plugin and run inference on VPU.
* `CPU` to use CPU plugin and run inference on ARM.

#### Get reference results on HOST

Run the following commands on HOST to generate reference results for KMB target network tests:

```bash
export IE_KMB_TESTS_DUMP_PATH=$KMB_PLUGIN_HOME/tests-dump
mkdir -p $IE_KMB_TESTS_DUMP_PATH
$DLDT_HOME/bin/intel64/Release/KmbFunctionalTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
rsync -avz $IE_KMB_TESTS_DUMP_PATH root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

#### Run tests on KMB Board

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
export IE_KMB_TESTS_DUMP_PATH=$KMB_WORK_DIR/tests-dump
$KMB_WORK_DIR/Release/KmbFunctionalTests --gtest_filter=*Kmb*NetworkTest*INT8_Dense*
```

### KMB plugin tests

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
$KMB_WORK_DIR/Release/KmbBehaviorTests
$KMB_WORK_DIR/Release/KmbFunctionalTests
```

### CPU ARM plugin tests

Run the following commands on the KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
export LD_LIBRARY_PATH=$KMB_WORK_DIR/Release/lib
export DATA_PATH=$KMB_WORK_DIR/temp/validation_set/src/validation_set
export MODELS_PATH=$KMB_WORK_DIR/temp/models
$KMB_WORK_DIR/Release/MklDnnBehaviorTests
$KMB_WORK_DIR/Release/MklDnnFunctionalTests
```

### OMZ accuracy validation

Use instructions from [VPU Wiki Accuracy Checker].
For CPU ARM plugin replace `-td KMB` command line option with `-td CPU`.

## Miscellaneous

`IE_VPU_KMB_DUMP_INPUT_PATH` environment variable can be used to dump input files for debugging purposes.
The variable must contain path to any writable directory.
All input blobs will be written to `$IE_VPU_KMB_DUMP_INPUT_PATH/input-dump%d.bin`.

`SIPP_FIRST_SHAVE` environment variable can be used to specify the first shave to be used for SIPP preprocessing.
The variable must contain a positive integer from `0` to `12`.
The number of shaves is `16`, maximal number of pipelines is `2`, maximal number of shaves per pipeline is `2`, which makes `16 - 2 * 2 = 12`.

[DLDT Project]: https://gitlab-icv.inn.intel.com/inference-engine/dldt
[KMB Plugin Project]: https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin
[VPU Wiki Accuracy Checker]: https://wiki.ith.intel.com/display/VPUWIKI/Set+up+and+Run+Accuracy+checker+on+ARM
[Yocto SDK]: https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin/blob/master/README.md#yocto-sdk
