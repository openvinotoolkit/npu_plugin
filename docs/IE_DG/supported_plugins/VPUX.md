# VPUX Plugin

## Introducing VPUX Plugin

VPUX Plugin was developed for inference of neural networks on the supported Intel&reg; Neural VPU devices:

  * Intel&reg; Neural VPU (3400VE)
  * Intel&reg; Neural VPU (3700VE)
  * Intel&reg; Neural VPU (3720VE)
  * Intel&reg; Neural VPU (3800V)
  * Intel&reg; Neural VPU (3900V)

## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host               | OS (64-bit)                          |
| :---               | :---                                 |
| Raptor Lake (dVPU) | MS Windows* 10                       |
| Meteor Lake (iVPU) | Ubuntu* 20, MS Windows* 10           |

### Offline Compilation

To run inference using VPUX plugin, Inference Engine Intermediate Representation needs to be compiled for a certain VPU device. Sometimes, compilation may take a while (several minutes), so it makes sense to compile a network before execution. Compilation can be done by a tool called `compile_tool`. An example of the command line running `compile_tool`:
```
compile_tool -d VPUX.3700 -m model.xml -c vpu.config
```
Where `VPUX` is a name of the plugin to be used, `3700` defines a VPU platform to be used for compilation (Intel&reg; Neural VPU (3700VE)), `model.xml` - a model to be compiled, `vpu.config` (optional) is a text file with config options.

If the platform is not specified, VPUX Plugin tries to determine it by analyzing all available system devices:
```
compile_tool -d VPUX -m model.xml
```

If system doesn't have any devices and platform for compilation is not provided, you will get an error `No devices found - platform must be explicitly specified for compilation. Example: -d VPUX.3700 instead of -d VPUX.`

The table below contains VPU devices and corresponding VPU platform:

| VPU device                                    | VPU platform |
| :-------------------------------------------  | :----------- |
| Intel&reg; Neural VPU (3400VE)                |   3400       |
| Intel&reg; Neural VPU (3700VE)                |   3700       |
| Intel&reg; Neural VPU (3720VE)                |   3720       |
| Intel&reg; Neural VPU (3800V)                 |   3800       |
| Intel&reg; Neural VPU (3900V)                 |   3900       |

To compile without loading to the device set environment variable IE_VPUX_CREATE_EXECUTOR to 0:
```
export IE_VPUX_CREATE_EXECUTOR=0
```
This is a temporary workaround that will be replaced later.

### Inference

For inference you should provide device parameter (see the table `Supported Configuration Parameters` below). Here are the examples of the command line running `benchmark_app`:
```
benchmark_app -d VPUX -m model.xml
```
Run inference on any available VPU device
```
benchmark_app -d VPUX.3900 -m model.xml
```
Run inference on any available slice of Intel&reg; Neural VPU (3900V)
```
benchmark_app -d VPUX.3800.0 -m model.xml
```
Run inference on the first slice of Intel&reg; Neural VPU (3800V)

## Supported Configuration Parameters

The VPUX plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                      |
| :---                  | :---             | :---             | :---                                                                             |
| `LOG_LEVEL`                                                            |`LOG_LEVEL_NONE`/</br>`LOG_LEVEL_ERROR`/</br>`LOG_LEVEL_WARNING`/</br>`LOG_LEVEL_DEBUG`/</br>`LOG_LEVEL_TRACE`                          |`LOG_LEVEL_NONE`                                               |Set log level</br>for VPUX plugin                                                                                                                                                                                                                                                                                                                                                                                                       |
| `PERF_COUNT`                                                           | `YES`/`NO`                                                                                                                             |`NO`                                                           |Enable or disable</br>performance counter                                                                                                                                                                                                                                                                                                                                                                                               |
| `DEVICE_ID`                                                            | empty/</br> `3400[.[0-3]]`/</br> `3700[.[0-3]]`/</br> `3900[.[0-3]]`/</br> `3800[.[0-3]]`                                              | empty (auto detection)                                        |Device identifier</br>`platform.slice`                                                                                                                                                                                                                                                                                                                                                                                                  |
| `PERFORMANCE_HINT`                                                     | `THROUGHPUT`/`LATENCY`                                                                                                                 | `THROUGHPUT` (for the</br>benchmark app)                      |Profile which determines</br>the number of</br>DPU groups (tiles)</br>and the number</br>of inference requests</br>if none of them</br>is modified manually.</br>The default parameter</br>values for each</br>profile are documented</br>in the [Performance Hint:</br>Default Number of</br>DPU Groups and</br>Inference Requests]</br>(#performance-hint-default</br>-number-of-dpu</br>-groups-and-inference</br>-requests) section |
| `PERFORMANCE_HINT_NUM_REQUESTS`                                        | `[0-]`                                                                                                                                 | `1`                                                           |(Optional) property that</br>backs the (above)</br>Performance Hints by</br>giving additional information</br>on how many</br>inference requests the</br>application will be</br>keeping in flight</br>usually this value</br>comes from the actual</br>use-case (e.g.</br>number of video-cameras,</br>or other sources</br>of inputs)                                                                                                 |
| `NUM_STREAMS`                                                          | `1` (The only supported</br> number for currently</br> supported platforms.</br> FIXME: update in the future)                          | `1`                                                           |The number of executor</br>logical partitions                                                                                                                                                                                                                                                                                                                                                                                           |
| `VPUX_COMPILER_FORCE_HOST_PRECISION_LAYOUT_CONVERSION`                 | `YES`/`NO`                                                                                                                             | Platform-dependent:</br>`YES` for Windows,</br>`NO` for Linux |This option allows</br>to use host</br>based pre- and</br>post- processing.</br>Note: Not only</br>the preprocessing operations</br>that are present</br>in the nGraph model</br>are removed from IR,</br>but also the</br>ones introduced in</br>the compiler itself                                                                                                                                                                   |
| `VPUX_DPU_GROUPS`                                                      | `[0-4]` for architecture VPUX30XX,</br> `[0-2]` for architecture VPUX3XX  | `-1`                                                          |Number of DPU groups                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `VPUX_DMA_ENGINES`                                                     | `[0-1]` for architecture VPUX30XX,</br> `[0-2]` for architectures VPUX311X, VPUX37XX                                         | `-1`                                                          |Number of DMA engines                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `VPUX_COMPILATION_MODE`                                                | `ReferenceSW`/</br>`ReferenceHW`/</br>`DefaultHW`/</br>`ShaveCodeGen`                                                                  | empty                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `VPUX_COMPILATION_MODE_PARAMS`                                         | (vpux_plugin_path)/</br>src/</br>vpux_compiler/</br>include/</br>vpux/</br>compiler/</br>pipelines.hpp                                 | empty                                                         |Config for HW-mode's</br>pipeline                                                                                                                                                                                                                                                                                                                                                                                                       |
| `VPUX_COMPILER_TYPE`                                                   | `MLIR`/</br>`DRIVER`                                                                                                                   | 'MLIR' for DEVELOPER_BUILD,</br>`DRIVER` otherwise            |Type of VPU</br>compiler to be used</br>for compilation of</br>a network                                                                                                                                                                                                                                                                                                                                                                |
| `VPUX_PRINT_PROFILING`                                                 | `NONE`/</br>`TEXT`/</br>`JSON`                                                                                                         | `NONE`                                                        |`NONE` - do not print</br>profiling info;</br>`TEXT`,</br>`JSON` - print detailed profiling</br>info during inference</br>in requested format                                                                                                                                                                                                                                                                                           |
| `VPUX_PROFILING_OUTPUT_FILE`                                           | `< Path to the file that contains profiling output >`                                                                                  | empty                                                         |std::cout is used</br>if parameter value</br>was empty                                                                                                                                                                                                                                                                                                                                                                                  |
| `VPUX_PLATFORM`                                                        | `AUTO_DETECT`/</br>`VPU3400_A0`/</br>`VPU3400`/</br>`VPU3700`/</br>`VPU3800`/</br>`VPU3900`/</br>`VPU3720`/</br>`EMULATOR`     | `AUTO_DETECT`                                                        |This option allows</br>to specify device.</br>If specified device</br>is not available</br>then creating infer</br>request will throw</br>an exception.                                                                                                                                                                                                                                                                                 |
| `MODEL_PRIORITY`                                                       | empty/</br>`LATENCY`/</br>`THROUGHPUT`/</br>`CUMULTIVE_THROUGHPUT`                                                                     | empty                                                         |Defines what model</br>should be provided</br>with more performant</br>bounded resource first                                                                                                                                                                                                                                                                                                                                           |
| `VPUX_USE_ELF_COMPILER_BACKEND`                                        | `YES`/`NO`                                                                                                                             | `NO`                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `DDR_HEAP_SIZE_MB`                                                     | `[500-1500]`                                                                                                                           | `500`                                                         |DDR heap size in MB                                                                                                                                                                                                                                                                                                                                                                                                                     |


### Performance Hint: Default Number of DPU Groups and Inference Requests

The following table shows the default parameter values used when setting the `THROUGHPUT` performance hint profile:

| VPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 1                    | 8                               |
| 3720                | 2 (all of them)      | 4                               |

The default parameter values applied when using the `LATENCY` profile:

| VPU Platform        | Number of DPU Groups | Number of Inference Requests    |
| :---                | :---                 | :---                            |
| 3700                | 4 (all of them)      | 1                               |
| 3720                | 2 (all of them)      | 1                               |

# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
