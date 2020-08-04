# HDDL2 Plugin

## Prerequsites

### Common x86_64

* Ubuntu 18.04 long-term support (LTS), 64-bit
* Kernel 5.0.x, 5.3.x
* Kernel headers
* OpenVINO™ Toolkit

### Common ARM

* Last BKC

## Set up PCIe

1. Check that device is visible in the system with command:

    ```bash
    lspci -d:6240
    ```

Expected output:

    01:00.0 Multimedia video controller: Intel corporation Device 6240

2. Install XLink IA kernel (PCIe driver) package from BKC (kmb-xlink-pcie-host-driver-dkms_*.deb) with the following commands:

    ```bash
    sudo apt install dkms
    sudo dpkg -i kmb-xlink-pcie-host-driver-dkms*
    ```

3. Check result of package installing with command:

    ```bash
    ll /lib/modules/`uname -r`/updates/dkms
    ```

Expected output:

    drwxr-xr-x 2 root root     4096 апр  2 13:42 ./
    drwxr-xr-x 3 root root     4096 апр  1 18:53 ../
    -rw-r--r-- 1 root root    52413 апр  2 13:42 mxlk.ko
    -rw-r--r-- 1 root root    55949 апр  2 13:42 xlink.ko

4. Install compiled driver with the following commands:

    ```bash
    sudo modprobe mxlk
    sudo modprobe xlink
    ```

5. Check that device was enabled correctly with command:

    ```bash
    dmesg | tail
    ```

Expected output:

    [  123.649561] mxlk: loading out-of-tree module taints kernel.
    [  123.649664] mxlk: module verification failed: signature and/or required key missing - tainting kernel
    [  123.651047] mxlk 0000:01:00.0: enabling device (0000 -> 0002)
    [  162.461544] xlink-driver xlink-driver: KeemBay xlink v0.94
    [  162.461581] xlink-driver xlink-driver: Major = 236 Minor = 0
    [  162.461626] xlink-driver xlink-driver: Device Driver Insert...Done!!!

## Set up HDDL2 plugin on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

2. Set environment variables with commands:

    ```bash
    cd $INTEL_OPENVINO_DIR/deployment_tools/inference-engine/external/hddl_unite
    source ./env_host.sh
    ```

3. Run scheduler service with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_scheduler_service
    ```

## Set up HDDL2 plugin on ARM

1. Set environment variables with commands:

    ```bash
    cd /opt/intel
    source ./env_arm.sh
    ```

2. Run device service on EVM with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/hddl_device_service
    ```

## Final check

* Expected output on x86_64:

    ```bash
    [16:52:48.7836][1480]I[main.cpp:55] HDDL Scheduler Service is Ready!
    ```

* Expected output on ARM:

    ```bash
    [20:56:15.3854][602]I[main.cpp:42] Device Service is Ready!
    ```
