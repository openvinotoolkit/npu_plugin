# How to deploy VPUX Plugin build to KMB board

Deploy OpenVINO artifacts to the KMB board:

```bash
rsync -avz --exclude "*.a" $OPENVINO_HOME/bin/aarch64/Release root@$KMB_BOARD_HOST:$KMB_WORK_DIR/
```

Deploy OpenVINO dependencies to the KMB board (replace `<ver>` with actual latest version which were downloaded by OpenVINO CMake script):

```bash
rsync -avz $OPENVINO_HOME/inference-engine/temp/tbb_yocto/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $OPENVINO_HOME/inference-engine/temp/openblas_<ver>_yocto_kmb/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
rsync -avz $OPENVINO_HOME/inference-engine/temp/opencv_<ver>_yocto_kmb/opencv/lib/*.so* root@$KMB_BOARD_HOST:$KMB_WORK_DIR/Release/lib/
```

Mount the HOST `$OPENVINO_HOME/inference-engine/temp` directory to the KMB board as a remote SSH folder.
Run the following commands on the KMB board for this:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
mkdir -p $KMB_WORK_DIR/temp
sshfs <username>@<host>:$OPENVINO_HOME/inference-engine/temp $KMB_WORK_DIR/temp
```

**Note:** to unmount the HOST `$OPENVINO_HOME/inference-engine/temp` directory from the KMB board use the following command:

```bash
# ssh root@$KMB_BOARD_HOST from HOST
fusermount -u $KMB_WORK_DIR/temp
```

#### Booting firmware

Before running anything you need to boot a correct firmware. By default, vpu_nvr.bin from /lib/firmware is booted.
The repo contains its own version of  firmware (`$KMB_PLUGIN_HOME/artifacts/vpuip_2/vpu.bin`) which is in sync with a current state of repo. To boot vpu.bin from the repo follow:

1. `rsync -avz $KMB_PLUGIN_HOME/artifacts/vpuip_2/vpu.bin root@$KMB_BOARD_HOST:/lib/firmware/vpu_custom.bin`
2. Make sure that there are no running applications at the moment
3. `echo "vpu_custom.bin"  > /sys/devices/platform/soc/soc\:vpusmm/fwname`
4. You can start your application. vpu_custom.bin will be booted.
