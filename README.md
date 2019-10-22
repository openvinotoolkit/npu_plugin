# kmb-plugin

KMBPlugin for Inference Engine


## How to build
There are two variants to build KMBPlugin: build-script and manual.
But for both variants you must first of all to build Inference Engine in dldt with
script "dldt/inference-engine/build-after-clone.sh" or see instructions in "dldt/inference-engine/CONTRIBUTING.md".

## Build with help of script:
1. Clone kmb-plugin from repository: `git clone git@gitlab-icv.inn.intel.com:inference-engine/kmb-plugin.git`
2. Find bash-script "build_after_clone.sh" in the base directory of KMBPlugin and run it.
3. When build finishes its work check output for possible errors.
4. Then run script "run_tests_after_build.sh" to check that you have built KMBPlugin correctly.

## Manual build:
1. Create variables with path to base directories of kmb-plugin and dldt:
You could use such commands:
- Go to base dldt directory and make `DLDT_HOME` variable with command:
  `export DLDT_HOME=$(pwd)`

- Go to base kmb-plugin directory and make `KMB_PLUGIN_HOME` variable with command:
  `export KMB_PLUGIN_HOME=$(pwd)`


2. Install additional packages for KMBPlugin:

   * Swig
   * python3-dev
   * python-numpy
   * metis
     with command:

     `sudo apt install swig python3-dev python-numpy libmetis-dev libmetis5 metis`

3. Move to dldt base directory and make some building with commands.
   **Note:**  if you miss `-DCMAKE_BUILD_TYPE=Debug` then you will not be able to debug your code in kmb-plugin:

   ```bash
   cd $DLDT_HOME
   mkdir -p $DLDT_HOME/build
   cd $DLDT_HOME/build
   git submodule init
   git submodule update --recursive
   cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_PLUGIN_RPATH=ON -DCMAKE_BUILD_TYPE=Debug ..
   make -j8
   ```
4. Move to base directory of KMBPlugin and build it with commands:

   ```bash
   cd $KMB_PLUGIN_HOME
   export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler
   git submodule update --init --recursive
   mkdir -p $KMB_PLUGIN_HOME/build
   cd $KMB_PLUGIN_HOME/build
   cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
   make -j8
   ```

5. To check results of previous steps it is recommended to execute tests with the following commands:
   If you built Inference Engine with parameter "-DENABLE_PLUGIN_RPATH=ON" then go to command beginning with "export MCM_HOME..", otherwise enter these commands:

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/bin/intel64/Debug/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/inference-engine/temp/opencv_4.1.0_ubuntu18/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/inference-engine/temp/tbb/lib
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KMB_PLUGIN_HOME/thirdparty/vsi_cmodel/vpusmm/x86_64
   ```

   ```bash
   export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler
   cd $DLDT_HOME/bin/intel64/Debug/
   ./KmbBehaviorTests --gtest_filter=*Behavior*orrectLib*kmb*
   ./KmbFunctionalTests
   ```
   **Note:** Make sure you are using `/intel64/Debug/` directory for Debug build and `/intel64/Release/` for Release in scripts of this section.

   If you see that all enabled tests are passed then you may congratulate yourself with successful build of KMBPlugin.

## Cross build for Yocto

Cross build use Yocto SDK. You can install it with:

``` sh
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/stable/ww28.5/oecore-x86_64-aarch64-toolchain-1.0.sh && \
        chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
        ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
        rm oecore-x86_64-aarch64-toolchain-1.0.sh
```
1. Clone and build metis library:

```sh
(\
  source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
  mkdir /tmp/metis && wget -P ~/Downloads "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz" && \
  tar xvzf ~/Downloads/metis-5.1.0.tar.gz -C ~/Downloads/ && cd ~/Downloads/metis-5.1.0 && \
  make config prefix=/tmp/metis/ && make install \
)
```

2. Clone dldt:

```sh
(cd .. && git clone git@gitlab-icv.inn.intel.com:inference-engine/dldt.git)
```

3. Configure and build inference engine:

Run following command from temporary build folder (e.g. `dldt\build`):

```sh
(\
  mkdir -p ../dldt/build && \
  cd ../dldt/build && \
  source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
  cmake -DENABLE_TESTS=ON .. && \
  cmake --build . --parallel $(nproc) \
)
```

4. Clone kmb-plugin:

```sh
(cd .. && git@gitlab-icv.inn.intel.com:inference-engine/kmb-plugin.git)
```
5. (Optional) Build mcmCompiler to ARM. Need open new terminal (not using yoctoSDK)

```sh
(\
    cd kmb-plugin/thirdparty/movidius/mcmCompiler/build && \
    cmake .. && \
    cmake --build . --parallel $(nproc) && \
    rm -rf !(meta) && exit
)
```

6. Build kmb-plugin.

Run following command from temporary build folder (e.g. `kmb-plugin\build`):

```sh
(\
  source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
  cmake -DInferenceEngineDeveloperPackage_DIR=$(realpath ../../dldt/inference-engine/build) \
  -DENABLE_MCM_COMPILER=ON -DMETIS_DIR=/tmp/metis/ .. && \
  cmake --build . --parallel $(nproc) \
)
```

## Testing on x86

You can run tests with inference using x86 platform with a fake device.
It can be done by a library called vpualModel. This library implements `ioctl` function,
which can be loaded before loading real `ioctl`(using `LD_PRELOAD`) to fake
a real device.

To be able to do it please follow the steps:

1. Create a dummy file for the XLink device
```sh
sudo touch /dev/xlnk
sudo chmod 666 /dev/xlnk
```
2. Enable corresponding environment to use the model
 ```sh
export LD_PRELOAD=<path-to-lib-folder-with-ie-binaries>/libvpualModel.so
export IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE=NATIVE
 ```
3. Run tests with inference. Example:
 ```sh
./KmbFunctionalTests --gtest_filter=*compareInferenceOutputWithReference*/0*
 ```

## Misc.

`IE_VPU_KMB_DUMP_INPUT_PATH` environment variable can be used to dump input
files for debugging purposes. This variable must contain path to any
writable directory. All input blobs will be written to
`$IE_VPU_KMB_DUMP_INPUT_PATH/input-dump%d.bin`

`SIPP_FIRST_SHAVE` environment variable can be used to specify the first shave
to be used for SIPP preprocessing. This variable must contain a positive
integer from 0 to 12. The number of shaves is 16, maximal number of pipelines
is 2, maximal number of shaves per pipeline is 2, which makes 16 - 2 * 2 = 12
