## OpenVINO VPUX plugin - KMB bypass

### VPUX plugin - KMB bypass Prerequisites

#### x86_64 host

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 5.0.x, 5.3.x, 5.4.x (you can use [ukuu kernel manager] to easily update system kernel)
* Kernel headers

    ```bash
    ls /lib/modules/`uname -r`/build || sudo apt install linux-headers-$(uname -r)
    ```

#### Common ARM

* [BKC Configuration KMB] (use instructions from [VPU Wiki Install FIP KMB] and [VPU Wiki Install Yocto KMB])

### VPUX plugin - KMB bypass Manual build

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

### Set up PCIe for HDDLUnite KMB

1. Configure board (use instructions from [VPU Wiki Board Configure KMB])

2. Install PCIe XLink and HDDL drivers (use instructions from [VPU Wiki PCIe drivers KMB])

### Set up VPUX plugin - KMB bypass on ARM

1. Download last version of HDDLUnite package from [BKC configuration KMB] (`hddlunite-kmb_*.tar.gz`) with the following commands:

    ```bash
    mkdir -p ~/Downloads
    cd ~/Downloads
    wget <HDDLUnite package link>
    ```

   If wget doesn't work properly, use browser instead.

2. Stop the service with command:
    ```bash
    systemctl stop deviceservice
    ```

3. Rename current HDDLUnite directory if it exists with the following commands:

    ```bash
    ls /opt/intel/hddlunite &&
    mv /opt/intel/hddlunite /opt/intel/hddlunite_orig
    ```

4. Unpack HDDLUnite package with the following commands:

    ```bash
    cd ~/Downloads
    tar -xzf hddlunite-kmb_*.tar.gz -C /opt/intel
    ```

5. Copy original env.sh script with command:

    ```bash
    cp /opt/intel/hddlunite_orig/env.sh /opt/intel/hddlunite
    ```

6. Reboot the board

### Set up VPUX plugin - KMB bypass on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

2. Reboot the host

3. Set environment variables with commands:

    ```bash
    cd $KMB_PLUGIN_HOME/temp/hddl_unite
    source ./env_host.sh
    ```

4. Run scheduler service with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_scheduler_service
    ```

### Final check

* Expected output on x86_64:

    ```bash
    [16:40:44.4156][2229]I[DeviceManager.cpp:612] Set mode(bypass) on device by config.
    [16:40:44.4840][2229]I[main.cpp:64] HDDL Scheduler Service is Ready!
    ```

* Expected output on ARM:

    ```bash
    root@keembay:~# journalctl -f -u deviceservice
    ... (some logs)
    Oct 28 01:39:53 hddl2keembay hddl_device_service[1105]: [01:39:53.6320][1105]I[main.cpp:80] HDDL Device Service is Ready!
    Oct 28 01:39:53 hddl2keembay hddl_device_service[1105]: [01:39:53.6389][1138]I[ModeManager.cpp:48] hddl unite set bypass mode
    ... (some logs)
    ```

# Links
[ukuu kernel manager]: https://github.com/teejee2008/ukuu
[VPU Wiki Board Configure KMB]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1503496133#HowtosetupPCIeforHDDLUnite-Configureboard
[VPU Wiki Install FIP KMB]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+FIP+via+fastboot
[VPU Wiki Install Yocto KMB]: https://wiki.ith.intel.com/display/VPUWIKI/How+to+flash+Yocto+Image+to+EMMC+via+fastboot
[VPU Wiki PCIe drivers KMB]: https://wiki.ith.intel.com/pages/viewpage.action?pageId=1503496133#HowtosetupPCIeforHDDLUnite-InstallPCIeXLinkdriver
[BKC Configuration KMB]: https://wiki.ith.intel.com/display/VPUWIKI/HDDL2#HDDL2-Configuration
