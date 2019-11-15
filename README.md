# KMB plugin for Inference Engine

## How to build

There are two variants to build kmb-plugin: build-script and manual.

But for both variants you must first of all to build Inference Engine in dldt with
script "dldt/inference-engine/build-after-clone.sh" or see instructions in "dldt/inference-engine/CONTRIBUTING.md".

## Build with help of script

1. Clone kmb-plugin from repository: `git clone git@gitlab-icv.inn.intel.com:inference-engine/kmb-plugin.git`
2. Find bash-script `build_after_clone.sh` in the base directory of kmb-plugin and run it.
3. When build finishes its work check output for possible errors.
4. Then run script `run_tests_after_build.sh` to check that you have built kmb-plugin correctly.

## Manual build

1. Create variables with path to base directories of kmb-plugin and dldt. You could use such commands.
    * Go to base dldt directory and make `DLDT_HOME` variable with command:
      `export DLDT_HOME=$(pwd)`
    * Go to base kmb-plugin directory and make `KMB_PLUGIN_HOME` variable with command:
      `export KMB_PLUGIN_HOME=$(pwd)`

2. Install additional packages for kmb-plugin:
    * Swig
    * python3-dev
    * python-numpy
    with command: `sudo apt install swig python3-dev python-numpy libmetis-dev libmetis5`

3. Move to dldt base directory and make some building with commands.

   ```bash
   cd $DLDT_HOME
   git submodule update --init --recursive
   mkdir -p $DLDT_HOME/build
   cd $DLDT_HOME/build
   cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
   make -j8
   ```

   **Note:** if you miss `-DCMAKE_BUILD_TYPE=Debug` then you will not be able to debug your code in kmb-plugin.
   **Note:** you might use another name for build sub-folder.

4. Move to base directory of kmb-plugin and build it with commands:

   ```bash
   cd $KMB_PLUGIN_HOME
   git submodule update --init --recursive
   mkdir -p $KMB_PLUGIN_HOME/build
   cd $KMB_PLUGIN_HOME/build
   cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
   make -j8
   ```

   **Note:** you might use another name for build sub-folder.

5. To check results of previous steps it is recommended to execute tests with the following commands.

   ```bash
   cd $DLDT_HOME/bin/intel64/Debug/
   ./KmbBehaviorTests --gtest_filter=*Behavior*orrectLib*kmb*
   ./KmbFunctionalTests
   ```

   **Note:** Make sure you are using `/intel64/Debug/` directory for Debug build and `/intel64/Release/` for Release in scripts of this section.

   If you see that all enabled tests are passed then you may congratulate yourself with successful build of kmb-plugin.

## Cross build for Yocto

Cross build use Yocto SDK. You can install it with:

``` sh
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/stable/ww28.5/oecore-x86_64-aarch64-toolchain-1.0.sh && \
        chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
        ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
        rm oecore-x86_64-aarch64-toolchain-1.0.sh
```

**Note:** The following steps assumes that you have cloned the dldt and kmb-plugin repositories and setup environment variables as in previous section.

1. Configure and cross-compile inference-engine.

   ```bash
   source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux
   cd $DLDT_HOME
   git submodule update --init --recursive
   mkdir -p $DLDT_HOME/build
   cd $DLDT_HOME/build
   cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
   make -j8
   ```

2. Configure and cross-compile kmb-plugin.

   ```bash
   source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux
   cd $KMB_PLUGIN_HOME
   git submodule update --init --recursive
   mkdir -p $KMB_PLUGIN_HOME/build
   cd $KMB_PLUGIN_HOME/build
   cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
   make -j8
   ```

3. To cross-compile mcmCompiler for ARM, perform the following steps:

    1. Build kmb-plugin natively on the host as described in previous section.
    2. Add the following options to CMake command line for kmb-plugin build:
        `-DENABLE_MCM_COMPILER=ON -DMCM_COMPILER_EXPORT_FILE=<path-to-kmb-plugin-native-build-dir>/mcmCompilerExecutables.cmake`

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
