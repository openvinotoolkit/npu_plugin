# HDDL2 Plugin

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

## Set up HDDL2 plugin on x86_64

1. Create user group with the following commands:

    ```bash
    sudo addgroup users
    sudo usermod -a -G users `whoami`
    ```

## Set up HDDL2 plugin on ARM

1. Rename current HDDLUnite directory if it exists with the following commands:

    ```bash
    ls /opt/intel/hddlunite &&
    mv /opt/intel/hddlunite /opt/intel/hddlunite_orig
    ```

2. Unpack HDDLUnite package from BKC (hddlunite-kmb_*.tar.gz) with command:

    ```bash
    cd ~/Downloads
    tar -xzf hddlunite-kmb_*.tar.gz -C /opt/intel
    ```

3. Copy original env.sh script with command:

    ```bash
    cp /opt/intel/hddlunite_orig/env.sh /opt/intel/hddlunite
    ```

4. Reboot the board

5. Check if the hddl_device_service boots with command:

    ```bash
    systemctl status deviceservice
    ```

* Expected output:

    ```bash
    ctl status deviceservice
    * deviceservice.service - HDDL Device Service
        Loaded: loaded (/etc/systemd/system/deviceservice.service; enabled; vendor preset: enabled)
        Active: active (running) since Fri 2020-10-02 04:17:52 UTC; 1h 49min ago
    ```

## Run HDDL2 plugin on x86_64

1. Install compiled drivers with the following commands:

    ```bash
    sudo modprobe mxlk
    sudo modprobe xlink
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

* Expected output:

    ```bash
    [16:52:48.7836][1480]I[main.cpp:55] HDDL Scheduler Service is Ready!
    ```

4. Open another terminal, repeat step 2 and then set HddlUnite mode with command:

    ```bash
    ${KMB_INSTALL_DIR}/bin/SetHDDLMode -m bypass
    ```

* Expected output:

    ```bash
    [14:51:07.2439][3900]I[SetHDDLMode.cpp:68] Have set mode on device successfully
    ```
