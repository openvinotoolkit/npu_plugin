# NPU Plugin

## Introducing NPU Plugin

NPU Plugin was developed for inference of neural networks on the supported Intel&reg; Neural Processing Unit (NPU) devices:

  * Intel&reg; NPU (3700VE)
  * Intel&reg; NPU (3720VE)

## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host               | OS (64-bit)                          |
| :---               | :---                                 |
| Raptor Lake (dNPU) | MS Windows* 10                       |
| Meteor Lake (iNPU) | Ubuntu* 20, MS Windows* 10           |

### Offline Compilation

To run inference using NPU plugin, Inference Engine Intermediate Representation needs to be compiled for a certain NPU device. Sometimes, compilation may take a while (several minutes), so it makes sense to compile a network before execution. Compilation can be done by a tool called `compile_tool`. An example of the command line running `compile_tool`:
```
compile_tool -d NPU.3700 -m model.xml -c NPU.config
```
Where `NPU` is a name of the plugin to be used, `3700` defines a NPU platform to be used for compilation (Intel&reg; NPU (3700VE)), `model.xml` - a model to be compiled, `NPU.config` (optional) is a text file with config options.

If the platform is not specified, NPU Plugin tries to determine it by analyzing all available system devices:
```
compile_tool -d NPU -m model.xml
```

If system doesn't have any devices and platform for compilation is not provided, you will get an error `No devices found - platform must be explicitly specified for compilation. Example: -d NPU.3700 instead of -d NPU.`

The table below contains NPU devices and corresponding NPU platform:

| NPU device                             | NPU platform |
| :------------------------------------  | :----------- |
| Intel&reg; NPU (3700VE)                |   3700       |
| Intel&reg; NPU (3720VE)                |   3720       |

To compile without loading to the device set environment variable IE_NPU_CREATE_EXECUTOR to 0:
```
export IE_NPU_CREATE_EXECUTOR=0
```
This is a temporary workaround that will be replaced later.

### Inference

For inference you should provide device parameter (see the table `Supported Configuration Parameters` below). Here are the examples of the command line running `benchmark_app`:
```
benchmark_app -d NPU -m model.xml
```
Run inference on any available NPU device
```
benchmark_app -d NPU.3700 -m model.xml
```
Run inference on any available 3700VE device

## Supported Configuration Parameters

The NPU plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                                                                                      |
| :---                  | :---             | :---             | :---                                                                                                                                             |
| `LOG_LEVEL`                                                           |`LOG_LEVEL_NONE`/</br>`LOG_LEVEL_ERROR`/</br>`LOG_LEVEL_WARNING`/</br>`LOG_LEVEL_DEBUG`/</br>`LOG_LEVEL_TRACE`                          |`LOG_LEVEL_NONE`                                               |Set log level</br>for NPU plugin.</br>It can also</br>be set through</br>OV_NPU_LOG_LEVEL</br>environment variable.                                                                                                                                                                                                                                                                                                                                                                                                        |
| `PERF_COUNT`                                                          | `YES`/`NO`                                                                                                                             |`NO`                                                           |Enable or disable</br>performance counter                                                                                                                                                                                                                                                                                                                                                                                               |
| `DEVICE_ID`                                                           | empty/</br> `3700`/</br> `3720`                                                                                                        | empty (auto detection)                                        |Device identifier                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `PERFORMANCE_HINT`                                                    | `LATENCY`/<br/>`THROUGHPUT`/<br/>`CUMULATIVE_THROUGHPUT`                                                                               | `LATENCY` (for the</br>benchmark app)                         |Profile which determines</br>the number of</br>DPU groups (tiles)</br>and the number</br>of inference requests</br>if none of them</br>is modified manually.</br>The default parameter</br>values for each</br>profile are documented</br>in the <a href="#performance-hint-default-number-of-dpu-groups-and-inference-requests">Performance Hint:</br>Default Number of</br>DPU Groups and</br>Inference Requests</a></br> section     |
| `PERFORMANCE_HINT_NUM_REQUESTS`                                       | `[0-]`                                                                                                                                 | `1`                                                           |(Optional) property that</br>backs the (above)</br>Performance Hints by</br>giving additional information</br>on how many</br>inference requests the</br>application will be</br>keeping in flight</br>usually this value</br>comes from the actual</br>use-case (e.g.</br>number of video-cameras,</br>or other sources</br>of inputs)                                                                                                 |
| `NUM_STREAMS`                                                         | `1` (The only supported</br> number for currently</br> supported platforms.)                                                           | `1`                                                           |The number of executor</br>logical partitions                                                                                                                                                                                                                                                                                                                                                                                           |
| `NPU_DPU_GROUPS`                                                      | `[0-4]` for architecture 30XX,</br> `[0-2]` for architecture 37XX                                                                      | `-1`                                                          |Number of DPU groups                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `NPU_DMA_ENGINES`                                                     | `[0-1]` for architecture 30XX,</br> `[0-2]` for architectures 37XX                                                                     | `-1`                                                          |Number of DMA engines                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `NPU_COMPILATION_MODE`                                                | `ReferenceSW`/</br>`ReferenceHW`/</br>`DefaultHW`/</br>`ShaveCodeGen`                                                                  | empty                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `NPU_COMPILATION_MODE_PARAMS`                                         | (NPU_plugin_path)/</br>src/</br>NPU_compiler/</br>include/</br>NPU/</br>compiler/</br>pipelines.hpp                                    | empty                                                         |Config for HW-mode's</br>pipeline                                                                                                                                                                                                                                                                                                                                                                                                       |
| `NPU_COMPILER_TYPE`                                                   | `MLIR`/</br>`DRIVER`                                                                                                                   | 'MLIR' for DEVELOPER_BUILD,</br>`DRIVER` otherwise            |Type of NPU</br>compiler to be used</br>for compilation of</br>a network                                                                                                                                                                                                                                                                                                                                                                |
| `NPU_PRINT_PROFILING`                                                 | `NONE`/</br>`TEXT`/</br>`JSON`                                                                                                         | `NONE`                                                        |`NONE` - do not print</br>profiling info;</br>`TEXT`,</br>`JSON` - print detailed profiling</br>info during inference</br>in requested format                                                                                                                                                                                                                                                                                           |
| `NPU_PROFILING_OUTPUT_FILE`                                           | `< Path to the file that contains profiling output >`                                                                                  | empty                                                         |std::cout is used</br>if parameter value</br>was empty                                                                                                                                                                                                                                                                                                                                                                                  |
| `NPU_PLATFORM`                                                        | `AUTO_DETECT`/</br>`EMULATOR`/</br>`3700`/</br>`3720`                                                                                  | `AUTO_DETECT`                                                 |This option allows</br>to specify device.</br>If specified device</br>is not available</br>then creating infer</br>request will throw</br>an exception.                                                                                                                                                                                                                                                                                 |
| `MODEL_PRIORITY`                                                      | empty/</br>`LATENCY`/</br>`THROUGHPUT`/</br>`CUMULTIVE_THROUGHPUT`                                                                     | empty                                                         |Defines what model</br>should be provided</br>with more performant</br>bounded resource first                                                                                                                                                                                                                                                                                                                                           |
| `NPU_USE_ELF_COMPILER_BACKEND`                                        | `YES`/`NO`                                                                                                                             | `YES`                                                         |                                                                                                                                                                                                                                                                                       


### Performance Hint: Default Number of DPU Groups and Inference Requests

The following table shows the default parameter values used when setting the `THROUGHPUT` performance hint profile:

| NPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 1                    | 8                               |
| 3720                | 2 (all of them)      | 4                               |

The default parameter values applied when using the `LATENCY` profile:

| NPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 4 (all of them)      | 1                               |
| 3720                | 2 (all of them)      | 1                               |

# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)