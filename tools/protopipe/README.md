# Protopipe
Protopipe is the C++ tool powered by the OpenCV G-API framework for prototyping AI workloads, evaluating their performance, and ensuring correctness.

## Installation
Install the latest [OpenVINO Runtime](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html) or build it from [Sources](https://github.com/openvinotoolkit/openvino).
### Build OpenCV G-API with OpenVINO support:
1. Clone OpenCV repo:
    `git clone -b 4.8.0 https://github.com/opencv/opencv`
2. Move to OpenCV base directory and build with the following command:
    ```
    mkdir -p build && cd build
    source <OpenVINO-Runtime-Archive>/setupvars.sh
    cmake ../ -DBUILD_LIST=gapi -DCMAKE_BUILD_TYPE=Release -DWITH_OPENVINO=ON
    cmake --build . --config Release --target opencv_gapi --parallel
    ```
    If `OpenVINO` has been built from sources use `-DOpenVINO_DIR=<path>` to specify path to the `OpenVINO` build directory.
   
### Build Protopipe 
Move to `npu-plugin` base directory and make sure all submodules are updated:    
`git submodule update --init --recursive`     
Consider one of the possible options to build `Protopipe`:
- Option 1: Build as part of `npu-plugin`. **Note**: OpenVINO built from sources is mandatory for this option.
    ```
    mkdir -p build && cd build
    cmake ../ -DOpenCV_DIR=<path-to-opencv-build> -DOpenVINODeveloperPackage_DIR=<path-to-openvino-build>
    cmake --build . --config Release --target protopipe --parallel
    ```
- Option 2: Standalone build.       
	Build dependencies:    
	- yaml-cpp:
		```
		mkdir -p yaml-cpp_build cd && yaml-cpp_build
		cmake ../<npu-plugin>/thirdparty/yaml-cpp -DCMAKE_INSTALL_PREFIX=install
		cmake --build . --config Release --target install --parallel
		```
    - gflags:
		```
        git clone https://github.com/gflags/gflags
        cd gflags
		mkdir -p gflags_build cd && gflags_build
		cmake ../ -DCMAKE_INSTALL_PREFIX=install
		cmake --build . --config Release --target install --parallel
		```
	Build Protopipe:
    ```
    mkdir -b protopipe_build && cd protopipe_build
    cmake <npu-plugin>/tools/protopipe/                           \
        -DOpenCV_DIR=<path-to-opencv-build                         \
        -Dyaml_cpp_DIR=<yaml-cpp_build/install/lib/cmake/yaml-cpp> \
        -Dgflags_DIR=<gflags_build/install>                        \
    
    cmake --build . --config Release --target protopipe --parallel
    ``` 
	If `OpenVINO` has been built from sources use `-DOpenVINO_DIR=<path>` to specify path to the `OpenVINO` build directory.
       
    Make sure `opencv_*` libraries are visible by configuring:               
    - `%PATH%` for `Windows`: 
        ```
        set PATH=<path-to-opencv>\build\bin\Release\;%PATH%
        ```
    - `$LD_LIBRARY_PATH` for `Linux`:
        ```
        export LD_LIBRARY_PATH=<path-to-opencv>/build/lib/:$LD_LIBRARY_PATH
        ```

## Quick start
Protopipe is using the `yaml` config file for describing the workload to simulate. 
Let's consider the simple workload that runs two `OpenVINO` models in parallel with different frame rates.      
`config.yaml` content:
```
# Path to the directory contains models.
model_dir:
  local: C:\myfolder\models
  
# Specifies target device name for all models within workload. (e.g CPU, GPU, NPU, etc)
device_name: CPU

# Triggers two streams consist of single model in parallel from different threads.
# 1) Stream 0: model_A.xml is working during 3 seconds with 10 FPS rate (triggered every 100ms)
# 2) Stream 1: model_B.xml is working during 3 seconds with 30 FPS rage (triggered every ~33ms)
multi_inference:
- input_stream_list: # stream 0
  - network:
    - { name: model_A.xml }
  target_fps: 10
  exec_time_in_secs: 3
- input_stream_list: # stream 1
  - network:
    - { name: model_B.xml }
  target_fps: 30
  exec_time_in_secs: 3
```
Run the tool: `./protopipe -cfg config.yaml` and see the performance metrics for every `stream` in the following format:
```
stream 0: throughput: <number> FPS, latency: min: <number> ms, avg: <number> ms, max: <number> ms, frames dropped: <number>/<number>
stream 1: throughput: <number> FPS, latency: min: <number> ms, avg: <number> ms, max: <number> ms, frames dropped: <number>/<number>
```

## Protopipe config
Protopipe workload consists of multiple `streams` that are running in parallel from different threads.

Every `stream` defines:
1. `network` - The list of the models that will be run sequentially one after another. 
Every element of `networks` is a dictionary which supports the following keys:
	- `priority`  - Model priority (`string`). Possible values: `LOW`, `NORMAL`, `HIGH`. Default: `NORMAL`.
	- `config`  - Model config (`dict`). Follow the documentation for particular OpenVINO plugin to know supported options.
	- `input_data` - Input data to use for inference (`string`/`dict`).      
	   E.g: `input_data: input.bin` or `input_data: {layer_name1: input1.bin, layer_name2: input2.bin}`. If not provided, input data will be generated randomly based on model precision/layout.
	 - `output_data` - Output data to compare with. (`string`/`dict`).     
	   E.g: `output_data: output_bin` or `output_data: {layer_name1: output1.bin, layer_name2: output2.bin}`. If not provided, outputs from the first iteration will be used as reference.
	 - Layer attributes:      
	    `ip` - Input layer precision. (`CV_32F`, `CV_16F`, `CV_8U`)   
	    `op` - Output layer precision. (`CV_32F`, `CV_16F`, `CV_8U`)     
	    `il` - Input layer layout. (`NCHW`, `CHW`, etc)   
	    `ol` - Output layer layout. (`NCHW`, `CHW`, etc)       
	    `iml` - Input model layout (only applicable for OpenVINO 2.0 API)    
	    `iml` - Output model layout (only applicable for OpenVINO 2.0 API)                 
        Attribute can be specified either for particular layer or for all.                         
        E.g ( `il` will be applied for all input layers.)                         
        ```
        - { name: model.xml, ip: { layer_name_0: CV_32F, layer_name_1: CV_8U }, il: NCHW } 
        ```
       

3. `delay_in_us` (Optional) - An integer value which defines delay between models inferences in microseconds. Might be used to simulate pre/post processing between models. (default: 0)
4. `target_fps` - Limit stream fps. (E.g If value is `10` stream will be triggered every `1000 / 10 = 100ms`). `0` - means no limits.
5. `exec_time_in_secs` - An integer value that defines the time period in seconds during which the stream should be executed. (default: 60)
6. `iteration_count` (Optional) - The number of iterations that stream should perform (mutually exclusive with `exec_time_in_secs`)

Consider the example of running two parallel streams during 10 seconds and triggered every 100ms (10 FPS):                 
`stream 0`: Model_A.xml -> Model_B.xml -> Model_C.xml    
`stream 1`: Model_D.xml -> (delay: 1ms) -> Model_E.xml         
```
model_dir:
  local: C:\models # Path to the dir contains models in *.xml / *.onnx format.
device_name: CPU # Any available OpenVINO device: CPU,NPU,GPU.

multi_inference:
  - input_stream_list:
    - network:
      - { name: Model_A.xml, ip: FP32, op: FP32, il: NCHW, ol: NCHW }
      - { name: Model_B.xml, ip: FP32, op: FP32, il: NCHW, ol: NCHW }
      - { name: Model_C.xml, ip: FP32, op: FP32, il: NCHW, ol: NCHW }
    target_fps: 0
    exec_time_in_secs: 10
    - network:
      - { name: Model_D.xml, ip: FP32, op: FP32, il: NCHW, ol: NCHW }
      - { name: Model_E.xml, ip: FP32, op: FP32, il: NCHW, ol: NCHW }
    delay_in_us: 1000
    target_fps: 10
    exec_time_in_secs: 10
```
## Execution          
Protopipe has the following `CLI` options to configure the execution behaviour.                     

`--cfg <path>` - Path to configuration file that describes workload.       
`--drop_frames`- Drop frames if they come earlier than stream is completed. E.g if `stream` works with `target_fps: 10` (~`100ms` latency) but stream iteration takes `150ms` - the next iteration will be triggered only in `50ms` if option is enabled.           
`--pipeline` - Enable pipelined execution for all streams.                      
`--ov_api_1_0`- Use obsolete OpenVINO 1.0 API.                    
