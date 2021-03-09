# How to build VPUX Plugin

## Requirements
### x86_64 host
For now only Ubuntu 18.04 x86 hosts are supported.

### ARM
#### Yocto SDK

To build ARM64 code for KMB board the Yocto SDK is required. It can be installed with the following commands:
```bash
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/dev-test-image/YP3p1/oecore-x86_64-aarch64-toolchain-1.0.sh && \
    chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
    ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
    rm oecore-x86_64-aarch64-toolchain-1.0.sh
```

## Dependencies

1. Install OpenVINO build dependencies using [OpenVINO Linux Setup Script].

2. In case of CMake version is too low error use `sudo snap install --classic cmake` and binary will be `/snap/bin/cmake`

    For building ARM64 using cmake from the Yocto SDK is recommended.

3. To initialize all submodules (or reinitialize when branch changes) you have to execute `git submodule update --init --recursive` inside a repo dir

4. Also it could be required to manually initialize `git lfs pull` especially if you have error message:

    `lib**.so: file format not recognized; treating as linker script`

5. Boost library

    `sudo apt install libboost-all-dev`

For details about OpenVINO build please refer to [OpenVINO Build Instructions].

## Build for X86_64

The X86_64 build is needed to get reference results for the tests.

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    mkdir -p $OPENVINO_HOME/build-x86_64
    cd $OPENVINO_HOME/build-x86_64
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D ENABLE_TESTS=ON \
        -D ENABLE_FUNCTIONAL_TESTS=ON \
        ..
    make -j${nproc}
    ```

2. Move to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build-x86_64
    cd $KMB_PLUGIN_HOME/build-x86_64
    cmake \
        -D InferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build-x86_64 \
        ..
    make -j${nproc}
    ```

## Build for ARM64

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    ( \
        source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux ; \
        mkdir -p $OPENVINO_HOME/build-aarch64 ; \
        cd $OPENVINO_HOME/build-aarch64 ; \
        cmake \
            -D ENABLE_TESTS=ON \
            -D ENABLE_FUNCTIONAL_TESTS=ON \
            -D THIRDPARTY_SERVER_PATH="http://nnt-srv01.inn.intel.com/dl_score_engine/" \
            .. ; \
        make -j${nproc} ; \
    )
    ```

2. **Open new console** in order to get clean environment and avoid side effects caused by running of previous commands for cross-compilation. 
3. Go to [KMB Plugin Project] base directory and build it with commands:

    ```bash
    mkdir -p $KMB_PLUGIN_HOME/build-aarch64
    cd $KMB_PLUGIN_HOME/build-aarch64
    cmake \
        -D CMAKE_TOOLCHAIN_FILE=$KMB_PLUGIN_HOME/cmake/oecore.arm64.toolchain.cmake \
        -D InferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build-aarch64 \
        ..
    make -j${nproc}
    ```

**Note:** Please use custom CMake toolchain file for [KMB Plugin Project],
default approach with `environment-setup-aarch64-ese-linux` is not supported.

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[OpenVINO Linux Setup Script]: https://raw.githubusercontent.com/openvinotoolkit/openvino/master/install_build_dependencies.sh
[OpenVINO Build Instructions]: https://github.com/openvinotoolkit/openvino/wiki/BuildingCode
