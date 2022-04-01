## OpenVINO VPUX plugin - TBH bypass

### VPUX plugin - TBH bypass Prerequisites

#### x86_64 host

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 4.18.0-15-generic with headers

    ```bash
    sudo apt update
    sudo apt install -y linux-image-4.18.0-15-generic linux-headers-4.18.0-15-generic linux-modules-extra-4.18.0-15-generic
    ```

#### Common ARM

* [BKC Configuration TBH] (use instructions from [VPU Wiki Boot TBH] and [VPU Wiki Install Yocto TBH])

### VPUX plugin - TBH bypass Manual build

1. Move to [OpenVINO Project] base directory and build it with the following commands:

    ```bash
    mkdir -p $OPENVINO_HOME/build-x86_64
    cd $OPENVINO_HOME/build-x86_64
    cmake \
        -D ENABLE_TESTS=ON \
        -D ENABLE_BEH_TESTS=ON \
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
        -D ENABLE_HDDL2=ON \
        -D ENABLE_HDDL2_TESTS=ON \
        ..
    make -j${nproc}
    ```
