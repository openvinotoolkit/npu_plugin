# How to use neural network profiling

## Using InferenceManagerDemo

### Step 1: Add PERF_COUNT flag to compile_tool configuration file

```bash
cat $OPENVINO_HOME/bin/intel64/<BUILD_TYPE>/config_info_log_KMB_B0_MLIR_PERF
LOG_LEVEL LOG_INFO
VPUX_COMPILER_TYPE MLIR
PERF_COUNT YES
VPUX_COMPILATION_MODE_PARAMS dpu-profiling=true dma-profiling=true sw-profiling=true
```

Optionally it is possible to deactivate particular profiling engine by setting `false` in following config line.
By default all 3 engines are set to `true`.

```
VPUX_COMPILATION_MODE_PARAMS dpu-profiling=true dma-profiling=true sw-profiling=true
```

### Step 2: Compile network

```bash
cd $OPENVINO_HOME/bin/intel64/<BUILD_TYPE>/
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
cd $OPENVINO_HOME/bin/intel64/<BUILD_TYPE>/
./prof_parser \
-b $VPUIP_HOME/application/demo/InferenceManagerDemo/test.blob \
-p $VPUIP_HOME/application/demo/InferenceManagerDemo/profiling-0.bin \
-f json \
-o ~/prof.json
```

* `-f` flag stands for "format" and can be either `text` or `json`.
`json` means [Trace Event Format]
* `-o` is an output file, will use `stdout` if omitted.

## Using benchmark_app

### Linux: Run benchmark_app on the machine with VPU

```bash
cd $OPENVINO_HOME/bin/intel64/<BUILD_TYPE>/
./benchmark_app \
-m <path/to/xml/or/blob> \
-d VPUX \
-niter 1 \
-pc
```

* Use `-d VPUX.3700` to specify keembay or leave `-d VPUX` to rely on autodetection.

### Windows

#### Step 1: Prepare config in JSON format

```bash
{
"VPUX": { "VPUX_COMPILER_TYPE":"DRIVER", "VPUX_PLATFORM":"3700", "LOG_LEVEL": "LOG_INFO" }
}
```

#### Step 2: Run benchmark_app with the config

```powershell
cd OpenVINO\bin\intel64\<BUILD_TYPE>\
benchmark_app.exe -m <path\to\xml\or\blob> -load_config=<path\to\.config.json> -d VPUX -niter 1 -pc
```

* `-niter` sets number of iterations, feel free to set more.
* `-pc` enables profiling counters. Be aware, only layer-level profiling is supported in benchmark_app!

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
