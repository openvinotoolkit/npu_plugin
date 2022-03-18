# How to use neural network profiling

## Using InferenceManagerDemo

### Step 1: Add PERF_COUNT flag to compile_tool configuration file

```bash
cat $OPENVINO_HOME/bin/intel64/Debug/config_info_log_KMB_B0_MLIR_PERF
LOG_LEVEL LOG_INFO
VPUX_COMPILER_TYPE MLIR
PERF_COUNT YES
```

### Step 2: Compile network

```bash
cd $OPENVINO_HOME/bin/intel64/Debug/
./compile_tool -c config_info_log_KMB_B0_MLIR_PERF \
-m ~/mobilenet/mobilenet.xml \
-d VPUX.3700 \
-o $VPUIP_HOME/application/demo/InferenceManagerDemo/mobilenet.blob \
-ip u8 \
-op fp32 \
-il NHWC
```

### Step 3: Run inference

```bash
cd $VPUIP_HOME/application/demo/InferenceManagerDemo/
ln -s mobilenet.blob test.blob
ln -s your_image.bin input-0.bin
make run CONFIG_FILE=.config srvIP=127.0.0.1 srvPort=30001 SOC_REVISION=B0 CONFIG_BLOB_BUFFER_MAX_SIZE_MB=100
```

### Step 4: Make sure profiling output appeared

```bash
ls profiling*
profiling-0.bin
```

### Step 5: Decode the output

```bash
cd $OPENVINO_HOME/bin/intel64/Debug/
./prof_parser \
-b $VPUIP_HOME/application/demo/InferenceManagerDemo/test.blob \
-p $VPUIP_HOME/application/demo/InferenceManagerDemo/profiling-0.bin \
-f json \
-o ~/prof.json
```

* `-f` flag stands for "format" and can be either `text` or `json`.
`json` means [Trace Event Format]
* `-o` is an output file, will use `stdout` if omitted.

## Using Yocto and single-image-test

### Step 1: Copy the network to your VPU device

```bash
scp -v mobilenet-v3-small* root@<your.kmb.ip.address>:/data/mobilenet
```

### Step 2: Add configuration flags

```bash
# In keembay shell
cat single-image-test-config.conf
VPUX_COMPILER_TYPE MLIR
VPUX_INFERENCE_TIMEOUT 0
PERF_COUNT YES
VPUX_PRINT_PROFILING JSON
VPUX_PROFILING_OUTPUT_FILE mobnet_prof_output.json
```

* `PERF_COUNT` tells compiler to compile blob with profiling enabled
* `VPUX_PRINT_PROFILING` output format, can be either `TEXT` or `JSON`
* `VPUX_PROFILING_OUTPUT_FILE` optional parameter - output file name

### Step 3: Run inference on the board

```bash
# In keembay shell
/data/Release/single-image-test \
--network /data/mobilenet/mobilenet-v3-small-1.0-224.xml \
--input /data/224x224/cat3.bmp \
--ip U8 \
--op FP32 \
--device VPUX \
--config /data/Release/single-image-test-config.conf

```

### Step 4: Copy output to your main machine

```bash
# In keembay shell
scp -v mobnet_prof_output.json user@<your.work.comp.ip>:/home/user
# Or alternative way - just copy-paste from that file:
cat mobnet_prof_output.json
```

## Last step

Open JSON file in any tool that supports [Trace Event Format]:

1. [Perfetto UI](https://ui.perfetto.dev/)
2. `chrome://tracing/` page in Google Chrome

[Trace Event Format]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#!
