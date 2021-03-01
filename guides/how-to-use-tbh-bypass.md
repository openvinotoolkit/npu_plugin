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

### Set up PCIe for HDDLUnite TBH

Install XLink and Secure XLink packages (use instructions from [VPU Wiki PCIe drivers TBH])

### Set up VPUX plugin - TBH bypass on ARM

Use instructions from [VPU Wiki Bypass TBH]

### Set up VPUX plugin - TBH bypass on x86_64

Use instructions from [VPU Wiki Bypass TBH]

### Final check

Use instructions from [VPU Wiki Bypass TBH Check]


# Links
[BKC Configuration TBH]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1700643473#ThunderBayHarbor-Configuration
[VPU Wiki Boot TBH]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+boot+up+Thunder+Bay+Harbor+board
[VPU Wiki Install Yocto TBH]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+Yocto+to+Thunder+Bay+Board
[VPU Wiki PCIe drivers TBH]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1710893290#Howtosetupby-passmodeforThunderBayHarbor-Pre-requisites
[VPU Wiki Bypass TBH]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+set+up+by-pass+mode+for+Thunder+Bay+Harbor
[VPU Wiki Bypass TBH Check]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+run+benchmark_app+in+by-pass+mode+for+TBH
