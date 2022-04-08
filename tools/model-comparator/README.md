# model-comparator tool

## Summary

The tool is used to perform per-layer comparison with reference device (CPU in FP32 mode, by default).
The tool cuts the network layer-by-layer and run inference on the sub-network to get intermediate results.
Then it performs per-element comparison with reference results and creates final HTML report with overall statistics.

## Prerequsites

The tool works only with IR v10, older versions are not supported.

The tool is built along with KMB plugin (see its README for instructions), but requires additional libraries:

* Boost
  * It can be installed with `sudo apt install libboost-all-dev` command on Ubuntu 18.04 x86 host.
  * It is already included into Yocto SDK for AARCH64.
* OpenCV
  * Comes with DLDT.
* gflags
  * Comes with DLDT.

## Usage

The tool is expected to be run in 2 stages:

* The first it is run on x86 host to compile the sub-networks and compute reference results.
* The second it is run on KMB board with the results of previous run to get actual results and build final comparison report.

The tool has the following required command line arguments:

* `--network_name <file name without extension>` - the name of IR file without `.xml` extension:
  * This option must be passed to both runs with the same value.
* `input_dir <path to directory>` - the path to input directory:
  * On x86 it will be the directory, where network IR file is located.
  * On KMB board it will be the directory with the results of x86 run.
* `output_dir <path to directory>` - the path to output directory:
  * On x86 the tool will dump sub-networks and reference results to that directory (and the directory should be synchronized with KMB board).
  * On KMB board the tool will create HTML report in this directory.
* `input_file <path to file>` - the path to input file (only images are supported for now):
  * It should be the same file for both x86 and KMB board runs.
* `--run_compile` - flag to turn on network compilation:
  * It must be used for x86 run only.
* `--run_ref` - flag to turn on reference results computation:
  * It must be used for x86 run only.
* `--run_infer` - flag to turn on inference and actual device:
  * It must be used for KMB board run only.

The tool has the following optional command line arguments:

* `--input_precision <precision>` - precision for the network input. Supported values: `U8`, `FP16`, `FP32`.
* `--input_layout <layout>` - layout for the network input. Supported values: `NCHW`, `NHWC`, `NCDHW`, `NDHWC`.
* `--output_precision <precision>` - precision for the network output. Supported values: `FP16`, `FP32`.
* `--output_layout <layout>` - layout for the network output. Supported values: `NCHW`, `NHWC`, `NCDHW`, `NDHWC`.
* `--black_list <comma separated list>` - the list of network layers to exclude from analysis:
  * Both layer name and layer type can be used.
* `--white_list <comma separated list>` - the list of network layers to include into analysis (only those layers will be used):
  * Both layer name and layer type can be used.

The tool has the following debug command line arguments:

* `--ref_device <device name>` - the device for reference generation (`CPU` be default).
* `--actual_device <device name>` - the device for actual inference (`KMB` be default).
* `--raw_export` - flag to use RAW export API (to generate compiled blobs compatible with other tools).
* `--log_level <level>` - log level for InferenceEngine library and plugins.

## Example

This is an example of the tool usage. It has the following assumptions:

* The KMB board was setup and is available via ssh
* DLDT and KMB plugin projects was built for both x86 and AARCH64 (see root README file for the instructions).

### Environment variables

The following environment variables should be set:

* The `DLDT_HOME` environment variable to the DLDT cloned directory on x86 side.
* The `KMB_PLUGIN_HOME` environment variable to the KMB plugin cloned directory on x86 side.
* The `KMB_BOARD_HOST` environment variable to the hostname or ip addess of the KMB board.
* The `X86_HOST` environment variable to the hostname or ip addess of the x86 host.
* The `X86_USER_NAME` environment variable to the x86 user name.

### x86 run

Run the following command:

```bash
mkdir -p ~/model-comparator
$DLDT_HOME/bin/inte64/Release/model-comparator \
    --run_compile \
    --run_ref \
    --network_name mobilenet-v2-pytorch-from-icv-bench-cache \
    --input_dir $DLDT_HOME/inference-engine/temp/models/src/models/KMB_models/INT8/public/MobileNet_V2 \
    --output_dir ~/model-comparator \
    --input_file $DLDT_HOME/inference-engine/temp/validation_set/src/validation_set/224x224/husky.bmp \
    --white_list 315,321,323 \
    --input_precision U8 \
    --input_layout NHWC
```

Expected output:

```
Parameters:
    Network name: mobilenet-v2-pytorch-from-icv-bench-cache
    Input directory: /home/user/dldt/inference-engine/temp/models/src/models/KMB_models/INT8/public/MobileNet_V2
    Output directory: /home/user/model-comparator
    Input file: /home/user/dldt/inference-engine/temp/validation_set/src/validation_set/224x224/husky.bmp
    Reference device: CPU
    Actual device: KMB
    Black list:
    While list: 315,321,323
    Run compile:1
    Run ref:1
    Run infer:0
    Raw export: 0
    Input precision: U8
    Input layout: NHWC
    Output precision:
    Output layout:
    Log level:

Load base network mobilenet-v2-pytorch-from-icv-bench-cache.xml

Load input file /home/user/dldt/inference-engine/temp/validation_set/src/validation_set/224x224/husky.bmp

Build sub-network up to layer 315
    Compile sub-network for KMB
    Calc reference with CPU

Build sub-network up to layer 321
    Compile sub-network for KMB
    Calc reference with CPU

Build sub-network up to layer 323
    Compile sub-network for KMB
    Calc reference with CPU
```

This run will copy the original IR to `~/model-comparator` directory and create the following directories:

* `~/model-comparator/sub-networks` - this directory will contain the per-layer sub-networks with compiled graphs for KMB.
* `~/model-comparator/blobs` - this directory will contain reference results in binary form.

### KMB board run

First we need to synchronize artifacts from x86 run with KMB board.
In this example it will be done with using remote SSH folder mounting.

Run the following commands on KMB board to mount x86 directories:

```bash
# ssh root@$KMB_BOARD_HOST from x86 host
scp $X86_USER_NAME@$X86_HOST:$KMB_PLUGIN_HOME/thirdparty/movidius/vpuip_2/vpu.bin /lib/firmware/vpu_custom.bin
mkdir -p /home/root/dldt-bin
sshfs $X86_USER_NAME@$X86_HOST:$DLDT_HOME/bin/aarch64 /home/root/dldt-bin
mkdir -p /home/root/dl-sdk-temp
sshfs $X86_USER_NAME@$X86_HOST:$DLDT_HOME/inference-engine/temp /home/root/dl-sdk-temp
mkdir -p /home/root/model-comparator
sshfs $X86_USER_NAME@$X86_HOST:/home/$X86_USER_NAME/$X86_USER_NAME/model-comparator /home/root/model-comparator
```

**Note:** to unmount the host directories use the following command:

```bash
# ssh root@$KMB_BOARD_HOST from x86 host
fusermount -u <mounted directory>
```

After that prepare the environment and run the tool on KMB board:

```bash
# ssh root@$KMB_BOARD_HOST from x86 host
export LD_LIBRARY_PATH=/home/root/dldt-bin/Release/lib:/home/root/dl-sdk-temp/opencv_4.2.0_yocto_kmb/opencv/lib:/home/root/dl-sdk-temp/tbb_yocto/lib:/home/root/dl-sdk-temp/openblas_0.3.7_yocto_kmb/lib
export VPU_FIRMWARE_FILE=vpu_custom.bin
/home/root/dldt-bin/model-comparator \
    --run_infer \
    --network_name mobilenet-v2-pytorch-from-icv-bench-cache \
    --input_dir /home/root/model-comparator \
    --output_dir /home/root/model-comparator \
    --input_file /home/root/dl-sdk-temp/validation_set/src/validation_set/224x224/husky.bmp \
    --white_list 315,321,323 \
    --input_precision U8 \
    --input_layout NHWC
```

Expected output:

```
Parameters:
    Network name: mobilenet-v2-pytorch-from-icv-bench-cache
    Input directory: /home/root/model-comparator
    Output directory: /home/root/model-comparator
    Input file: /home/root/dl-sdk-temp/validation_set/src/validation_set/224x224/husky.bmp
    Reference device: CPU
    Actual device: KMB
    Black list:
    While list: 315,321,323
    Run compile:0
    Run ref:0
    Run infer:1
    Raw export: 0
    Input precision: U8
    Input layout: NHWC
    Output precision:
    Output layout:
    Log level:

Load base network mobilenet-v2-pytorch-from-icv-bench-cache.xml

Load input file /home/root/dl-sdk-temp/validation_set/src/validation_set/224x224/husky.bmp

Build sub-network up to layer 315
    Import sub-network for KMB
Compiled on Mar  5 2020 at 06:23:10
Booting VPUAL dispatcher...

Callback signal handler installed

Opening Device File /dev/xlnk
Started VPU with status: 0
Dispatcher channel opened successfully.
Opened XLink channel - 11
Opened XLink channel - 12
Opened XLink channel - 13
Opened XLink channel - 14
    Import reference
    Run infer on KMB
    Compare with reference

Build sub-network up to layer 321
    Import sub-network for KMB
Opened XLink channel - 11
Opened XLink channel - 12
Opened XLink channel - 13
Opened XLink channel - 14
    Import reference
    Run infer on KMB
    Compare with reference

Build sub-network up to layer 323
    Import sub-network for KMB
Opened XLink channel - 11
Opened XLink channel - 12
Opened XLink channel - 13
Opened XLink channel - 14
    Import reference
    Run infer on KMB
    Compare with reference

Shuting down VPUAL dispatcher...
Stopped the VPU with custom firmware status: 0
```

The run will create the following directory on x86 host:

* `~/model-comparator/report` with per-layer comparison HTML report and visualized difference maps.
