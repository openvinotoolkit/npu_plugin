# VPUX plugin - KMB bypass

## Prerequsites

### Common x86_64

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 5.x
* Kernel headers
* OpenVINO™ Toolkit

### Common ARM

* Last BKC

## Set up PCIe

1. Check that device is visible in the system with command:

    ```bash
    lspci -d:6240
    ```

* Expected output:

    ```bash
    01:00.0 Multimedia video controller: Intel corporation Device 6240
    ```

2. Install XLink IA kernel (PCIe driver) package from BKC (kmb-xlink-pcie-host-driver-dkms_*.deb) with the following commands:

    ```bash
    sudo apt install dkms
    sudo dpkg -i kmb-xlink-pcie-host-driver-dkms*
    ```

3. Install Hddl IA kernel (PCIe driver) package from BKC (kmb-hddl-driver-dkms_*.deb) with command:

    ```bash
    sudo dpkg -i kmb-hddl-driver-dkms*
    ```

4. Check the package installation results with command:

    ```bash
    ll /lib/modules/`uname -r`/updates/dkms
    ```

* Expected output:

    ```bash
    drwxr-xr-x 2 root root  4096 окт  6 08:43 ./
    drwxr-xr-x 3 root root  4096 окт  6 08:32 ../
    -rw-r--r-- 1 root root 25016 окт  6 08:43 emc2103.ko
    -rw-r--r-- 1 root root 11344 окт  6 08:43 Gpio-asm28xx.ko
    -rw-r--r-- 1 root root 28928 окт  6 08:43 hddl_device_server.ko
    -rw-r--r-- 1 root root 10280 окт  6 08:43 intel_tsens_host.ko
    -rw-r--r-- 1 root root 58800 окт  6 08:32 mxlk.ko
    -rw-r--r-- 1 root root 74152 окт  6 08:32 xlink.ko
    -rw-r--r-- 1 root root 18408 окт  6 08:43 xlink-smbus.ko
    ```

5. Install compiled drivers with the following commands:

    ```bash
    sudo modprobe mxlk
    sudo modprobe xlink
    ```

6. Check that device was enabled correctly with command:

    ```bash
    dmesg | tail
    ```

* Expected output:

    ```bash
    [  123.649561] mxlk: loading out-of-tree module taints kernel.
    [  123.649664] mxlk: module verification failed: signature and/or required key missing - tainting kernel
    [  123.651047] mxlk 0000:01:00.0: enabling device (0000 -> 0002)
    [  162.461544] xlink-driver xlink-driver: KeemBay xlink v0.94
    [  162.461581] xlink-driver xlink-driver: Major = 236 Minor = 0
    [  162.461626] xlink-driver xlink-driver: Device Driver Insert...Done!!!
    ```
7. Install thermal monitor driver (thermaldaemon-*-x86_64.deb) from BKC with the following command:

    ```bash
    sudo dpkg -i thermaldaemon-*-x86_64.deb
    ```

## Set up VPUX plugin - KMB bypass on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

2. Reboot the host

## Run VPUX plugin - KMB bypass on x86_64

1. Set environment variables with the following commands:

    ```bash
    cd $INTEL_OPENVINO_DIR/deployment_tools/inference-engine/external/hddl_unite
    source ./env_host.sh
    ```

2. Run scheduler service with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_scheduler_service
    ```

* Expected output:

    ```bash
    [16:40:44.4156][2229]I[DeviceManager.cpp:612] Set mode(bypass) on device by config.
    [16:40:44.4840][2229]I[main.cpp:64] HDDL Scheduler Service is Ready!
    ```

# VPUX plugin - TBH bypass

## Prerequsites

### Common x86_64

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 4.18.0-15
* Kernel headers
* OpenVINO™ Toolkit

### Common ARM

* Last BKC

## Set up PCIe

1. Check that device is visible in the system with command:

    ```bash
    lspci -d:4fc0
    ```

* Expected output:

    ```bash
    1a:00.0 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.1 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.2 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.3 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.4 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.5 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.6 Multimedia video controller: Intel Corporation Device 4fc0
    1a:00.7 Multimedia video controller: Intel Corporation Device 4fc0
    ```

2. Install XLink package from BKC (xlink-pcie.zip) with the following commands:

    ```bash
    sudo mkdir ~/xlink
    unzip xlink-pcie.zip -d ~/xlink
    sudo cp ~/xlink/xlink-pcie/IA-HOST/libXLink.so /usr/lib
    ```

3. Install Secure XLink package from BKC (xlink-security-pcie.zip) with the following commands:

    ```bash
    unzip xlink-security-pcie.zip -d ~/xlink
    sudo cp ~/xlink/xlink-security-pcie/tigerlake/libSecureXLink.so /usr/lib
    ```

## Set up VPUX plugin - TBH bypass on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

2. Reboot the host

## Run VPUX plugin - TBH bypass

1. (ARM) Prepare XLink with the following commands:

    ```bash
    mount / -o rw,remount
    cd /sys/kernel/config/pci_ep

    for FUNC_NUM in 0 1 2 3 4 5 6 7 ; do \
    mkdir functions/mxlk_pcie_epf/func${FUNC_NUM} ; \
    ln -s functions/mxlk_pcie_epf/func${FUNC_NUM} controllers/82000000.pcie_ep/ ; \
    echo "1" > controllers/82000000.pcie_ep/start ; \
    done
    ```

2. (x86_64) Prepare XLink with the following commands:

    ```bash
    sudo rmmod mxlk
    sudo insmod ~/xlink/xlink-pcie/IA-HOST/IA-HOST-4.18.0-15-generic/mxlk.ko
    sudo insmod ~/xlink/xlink-pcie/IA-HOST/IA-HOST-4.18.0-15-generic/xlink.ko
    sudo chmod 777 /dev/xlnk
    ```

3. (x86_64) Set environment variables with the following commands:

    ```bash
    cd $INTEL_OPENVINO_DIR/deployment_tools/inference-engine/external/vpux_4/hddl_unite
    source ./env_host.sh
    ```

4. (x86_64) Run scheduler service with command:

    ```bash
    ${KMB_INSTALL_DIR}/vpux_4/bin/hddl_scheduler_service
    ```

5. (ARM) Run device services with the following commands:

    ```bash
    cd /opt/intel/hddlunite
    source ./env.sh
    cd $KMB_INSTALL_DIR/bin
    ./hddl_device_service 0 &
    ./hddl_device_service 1 &
    ./hddl_device_service 2 &
    ./hddl_device_service 3 &
    ```

6. (x86_64) Set bypass mode with the following commands (open another terminal):

    ```bash
    cd $INTEL_OPENVINO_DIR/deployment_tools/inference-engine/external/vpux_4/hddl_unite
    source ./env.sh
    cd $KMB_INSTALL_DIR/bin
    ./SetHDDLMode -m bypass
    ```

* Expected output:
    - hddl_scheduler_service reports that it is ready;
    - hddl_device_services report that they are ready;
    - SetHDDLMode reports that mode was set successfully.
