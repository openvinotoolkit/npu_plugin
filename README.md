# OpenVINO KMB plugin

## Prerequsites

### Yocto SDK

To build ARM64 code for KMB board the Yocto SDK is required. It can be installed with the following commands:

```bash
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/development/20191011-1106/oecore-x86_64-aarch64-toolchain-1.0.sh && \
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
            -D MCM_COMPILER_EXPORT_FILE=$KMB_PLUGIN_HOME/build-x86_64/mcmCompilerExecutables.cmake \
            .. ; \
        make -j8 ; \
    )
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

## Testing on x86_64

You can run tests with inference using x86 platform with a fake device.
It can be done by a library called `vpualModel` library.
The library implements `ioctl` function, which can be loaded before loading real `ioctl` (using `LD_PRELOAD`) to fake a real device.

To be able to do it please follow the steps:

1. Create a dummy file for the XLink device:

    ```bash
    sudo touch /dev/xlnk
    sudo chmod 666 /dev/xlnk
    ```

2. Enable corresponding environment to use the model:

    ```bash
    export LD_PRELOAD=$DLDT_HOME/bin/intel64/Release/lib/libvpualModel.so
    export IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE=NATIVE
    ```

3. Run tests with inference. Example:

    ```bash
    $DLDT_HOME/bin/intel64/Release/KmbBehaviorTests
    $DLDT_HOME/bin/intel64/Release/KmbFunctionalTests
    ```

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
