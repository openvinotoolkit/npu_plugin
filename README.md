# KMB plugin for Inference Engine

## How to build

There are two variants to build kmb-plugin: build-script and manual.

But for both variants you must first of all to build Inference Engine in dldt with
script `dldt/inference-engine/build-after-clone.sh` or see instructions in `dldt/inference-engine/CONTRIBUTING.md`.

## Build with help of script

1. Clone kmb-plugin from repository: `git clone git@gitlab-icv.inn.intel.com:inference-engine/kmb-plugin.git`
2. Find bash-script `build_after_clone.sh` in the base directory of kmb-plugin and run it.
3. When build finishes its work check output for possible errors.
4. Then run script `run_tests_after_build.sh` to check that you have built kmb-plugin correctly.

## Manual build

### Common setup

1. Create variables with path to base directories of kmb-plugin and dldt. You could use such commands.
    * Go to base dldt directory and make `DLDT_HOME` variable with command: `export DLDT_HOME=$(pwd)`
    * Go to base kmb-plugin directory and make `KMB_PLUGIN_HOME` variable with command: `export KMB_PLUGIN_HOME=$(pwd)`

2. Update git sub-modules in both dldt and kmb-plugin:

    ```bash
    cd $DLDT_HOME
    git submodule update --init --recursive
    cd $KMB_PLUGIN_HOME
    git submodule update --init --recursive
    ```

### Native build

1. Move to dldt base directory and build it with commands:

    ```bash
    mkdir -p $DLDT_HOME/build
    cd $DLDT_HOME/build
    cmake \
        -D ENABLE_TESTS=ON \
        -D ENABLE_BEH_TESTS=ON \
        -D ENABLE_FUNCTIONAL_TESTS=ON \
        ..
    make -j8
    ```

2. Move to base directory of kmb-plugin and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build
    cd $KMB_PLUGIN_HOME/build
    cmake \
        -D InferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build \
        ..
    make -j8
    ```

3. To check results of previous steps it is recommended to execute tests with the following commands.

    ```bash
    cd $DLDT_HOME/bin/intel64/Debug/
    ./KmbBehaviorTests --gtest_filter=*Behavior*orrectLib*kmb*  // no valid gtest-filter anymore
    ./KmbFunctionalTests
    ```

    **Note:** Make sure you are using `/intel64/Debug/` directory for Debug build and `/intel64/Release/` for Release in scripts of this section.

    If you see that all enabled tests are passed then you may congratulate yourself with successful build of kmb-plugin.

### Cross build for Yocto

Cross build use Yocto SDK. You can install it with:

```bash
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/stable/ww28.5/oecore-x86_64-aarch64-toolchain-1.0.sh && \
        chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
        ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
        rm oecore-x86_64-aarch64-toolchain-1.0.sh
```

1. Build mcmCompiler for **x86** (needed for cross-compilation).

    **Note:** Skip this step, if you don't want to cross-compile mcmCompiler for ARM.

    **Note:** Skip this step, if you have full **x86** build of kmb-plugin as described in previous section.

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/mcmCompiler-build-x86
    cd $KMB_PLUGIN_HOME/mcmCompiler-build-x86
    cmake \
        -D MCM_COMPILER_BUILD_PYTHON=OFF \
        -D MCM_COMPILER_BUILD_TESTS=OFF \
        -D MCM_COMPILER_BUILD_EXAMPLES=OFF \
        ../thirdparty/movidius/mcmCompiler
    make flatc gen_composition_api -j8
    ```

2. Configure and cross-compile inference-engine for ARM.

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $DLDT_HOME/build-aarch64 ; \
        cd $DLDT_HOME/build-aarch64 ; \
        cmake \
            -D ENABLE_TESTS=ON \
            -D ENABLE_BEH_TESTS=ON \
            -D ENABLE_FUNCTIONAL_TESTS=ON \
            .. ; \
        make -j8 ; \
    )
    ```

3. Configure and cross-compile kmb-plugin.

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $KMB_PLUGIN_HOME/build-aarch64 ; \
        cd $KMB_PLUGIN_HOME/build-aarch64 ; \
        cmake \
            -D InferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build-aarch64 \
            -D MCM_COMPILER_EXPORT_FILE=../mcmCompiler-build-x86/mcmCompilerExecutables.cmake \
            .. ; \
        make -j8 ; \
    )
   ```

    **Note:** If you don't want to cross-compile mcmCompiler for ARM,
      remove `MCM_COMPILER_EXPORT_FILE` from configure command.

    **Note:** If you have full **x86** build of kmb-plugin use
      `-D MCM_COMPILER_EXPORT_FILE=<kmb-plugin-x86-build-dir>/mcmCompilerExecutables.cmake` option.

## Testing on x86

You can run tests with inference using x86 platform with a fake device.
It can be done by a library called vpualModel. This library implements `ioctl` function,
which can be loaded before loading real `ioctl`(using `LD_PRELOAD`) to fake
a real device.

To be able to do it please follow the steps:

1. Create a dummy file for the XLink device

    ```bash
    sudo touch /dev/xlnk
    sudo chmod 666 /dev/xlnk
    ```

2. Enable corresponding environment to use the model

    ```bash
    export LD_PRELOAD=<path-to-lib-folder-with-ie-binaries>/libvpualModel.so
    export IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE=NATIVE
    ```

3. Run tests with inference. Example:

    ```bash
    ./KmbFunctionalTests --gtest_filter=*compareInferenceOutputWithReference*/0*
    ```

## Misc

`IE_VPU_KMB_DUMP_INPUT_PATH` environment variable can be used to dump input
files for debugging purposes. This variable must contain path to any
writable directory. All input blobs will be written to
`$IE_VPU_KMB_DUMP_INPUT_PATH/input-dump%d.bin`

`SIPP_FIRST_SHAVE` environment variable can be used to specify the first shave
to be used for SIPP preprocessing. This variable must contain a positive
integer from 0 to 12. The number of shaves is 16, maximal number of pipelines
is 2, maximal number of shaves per pipeline is 2, which makes 16 - 2 * 2 = 12
