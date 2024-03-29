# NPU Plugin

## Introduction &nbsp;

This is the OpenVINO Plugin for Intel&reg; Neural Processing Unit (NPU) devices.

&nbsp;
## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host                         | NPU device  | OS (64-bit)                          |
| :---                         | :---        | :---                                 |
| Raptor Lake (discrete   NPU)   | NPU 3700    | MS Windows* 11                       |
| Meteor Lake (integrated NPU)   | NPU 3720    | Ubuntu* 20, MS Windows* 11           |


&nbsp;
## High Level Design

![High Level Design](./img/high_level_design.png)


&nbsp;
## Description

NPU Plugin is a software library that:
* Implements the unified OpenVINO Plugin API used to compile and execute neural networks on NPU devices.
* Depending on the type of compiler instantiated, it uses either the API exposed by the NPU compiler or the graph extension API exposed by the driver to convert the OpenVINO specific representation of the model into a proprietary format. The compiler performs platform specific optimizations in order to efficiently schedule the execution of layers and memory transactions on various NPU hardware submodules.
* Uses the Level Zero API implemented by the NPU user mode driver (UMD) to execute the model on the device.

The plugin library is included inside the OpenVINO package while the compiler is packaged inside UMD and released separately.

Note: Aligning with the platform and OpenVINO documentation, neural networks will be referred to with the more generic term of models in the rest of this document.

&nbsp;
## Supported Compilers

The compiler is based on the MLIR project. There are two different ways for the plugin to access and use the compiler:
* Compiler In Plugin: A library that will be dynamically loaded by the plugin to compile the model. This is intended to be used only for development purposes. Backward/Forward compatibility is not guaranteed for precompiled models.
* Compiler In Driver: A library included in UMD. The plugin will use the available graph extension APIs exposed by the driver to request the model compilation. Two additional adapters are needed to support the integration of the compiler inside the driver:

&nbsp;
## Supported Engine Backends

An engine backend is an abstraction layer on top of underlying APIs used to execute models. It is meant to include all the required functionality and infrastructure required to execute multiple models in parallel on one or multiple devices. Multiple engine backends are supported by the plugin:
* L0 (Level Zero) backend
* IMD backend

&nbsp;
## Model Compilation

NPU plugin implements the OpenVINO Core "compile_model" API that converts the model representation into a proprietary format that can be executed on the NPU device:

```
    ov::CompiledModel compiled_model = core.compile_model(model, "NPU" [, config]);
```


Release packages will use the compiler from driver by default. In case the library for compiler in plugin is also included in the build, this compiler type can be used for model compilation by setting the `ov::intel_vpux::compiler_type` property to `MLIR`. The default compiler type is `MLIR` when the plugin is built with DEVELOPER_BUILD=ON.

### Device management

NPU Plugin automatically detects the underlying platform by querying device properties from the NPU driver. Platform information is provided through configs to the NPU compiler to compile the model for that specific device.  
Offline compilation is supported only when "Compiler in Plugin" library is also included in the build and when `ov::intel_vpux::compiler_type` is set to `MLIR`.  
There are two ways to provide the platform information for offline compilation:
* Through the NPU_PLATFORM config. An extra config can be passed as the third argument to "compile_model". NPU_PLATFORM can be set to one of the supported platforms.
* Through the DEVICE_ID. Example "NPU.3720". This will be deprecated soon.

### Model caching

There are two important compilation related metrics when executing models on NPU devices:
* First Ever Inference Latency (FEIL): Measures all steps required to compile and execute a model on the device for the first time. It includes model compilation time, the time required to load and initialize the model on the device and the first inference execution.
* First Inference Latency (FIL): Measures the time required to load and initialize the pre-compiled model on the device and the first inference execution.


#### UMD dynamic model caching

UMD model caching is enabled by default in the current NPU driver to improve time to first inference (FIL). The model is stored in the cache after the compilation (included in FEIL) based on a hash key. The UMD generates the key from the input IR model and build arguments and then requests the DirectX Shader cache session to store the model with the computed key. Any subsequent request to compile the same IR model with the same arguments would cause the pre-compiled model to be read from the cache instead of being recompiled.

#### OpenVINO model caching

It is enabled when `ov::cache_dir` property is set and it is a common mechanism for all OpenVINO plugins. UMD model caching will be automatically bypassed by the NPU plugin when `ov::cache_dir` is set so the model will only be stored in the OpenVINO cache after the compilation. When a cache hit occurs for subsequent compilation requests, plugin will import the model instead of recompiling it.

More details about OpenVINO model caching can be found here: [Model Caching Overview](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Model_caching_overview.html).

### Compiler adapters

Two additional layers are required to support the compiler from driver:
* Compiler Adapter - It serializes the OpenVINO internal representation of the model (ov::model) into an in-memory IR that will be provided to the NPU driver  
* VCL - It deserializes the in-memory IR given by the NPU driver and prepares it for the compiler

The interface between plugin and driver is based on an in-memory IR to facilitate backward and forward compatibility between two software packages (OpenVINO and NPU driver) that inherently have a different release cadence.

&nbsp;
## Model Execution

NPU plugin will use the Level Zero (L0) API to execute the precompiled model on the NPU Device. The inference is executed as a standard L0 workload by describing the required tasks inside a command list and by submitting the list to the command queue for execution. The plugin will not use the CPU to execute any part of the inference workload. No pre/post processing workloads are executed on the CPU either, the entire inference will be offloaded on the NPU device.

### Device management

There is currently no support for multiple devices (N x discrete + integrated), one single level zero device will be enumerated during level zero backend initialization. Support for multiple devices will be added in future releases.

### Inference pipeline

The result of the model compilation is represented through a NetworkDescription. This model description is passed by the plugin to the driver to create a level zero graph instance and obtain a graph handle that can later be used to execute multiple inferences in parallel for the same model. Since the same model instance is shared across all subsequent inference objects, this initialization step is performed by default right after the model is compiled and it can be postponed until the creation of the first inference request through the use of an environment variable: "IE_NPU_CREATE_EXECUTOR" (IE_NPU_CREATE_EXECUTOR=0 to postpone the initialization).

Users can create one or more inference requests for a compiled model using OpenVINO API:

```
    ov::InferRequest request = compiled_model.create_infer_request();
```

One unique level zero command queue is currently used to execute all inference requests created for the same model.

&nbsp;
## Supported Properties

Properties can be used to query and adjust the behavior of the NPU plugin itself or various parameters that control model compilation and execution.  

The following methods are made available to return the value of a given property (at core level or model specific):
```
    plugin_properties = ov.get_property("NPU", <property_name>);
    [...]
    model_properties = compiled_model.get_property(<property_name>);
```

The following methods are made available to set the value of a given property (at core level or model specific):
```
    ov.set_property("NPU", {{Key, Value}});
    [...]
    compiled_model.set_property({{Key, Value}});
```

The following properties are supported:

| Parameter Name |            | Description | Supported Values | Default Value |
| :---           | :---       | :---        |:---              |:--            |
| `ov::supported_properties`/</br>`SUPPORTED_METRICS`/</br>`SUPPORTED_CONFIG_KEYS` | RO | Returns a list of all supported properties.</br> Can be queried on runtime. | `N/A` | `N/A` |
| `ov::caching_properties`/</br>`CACHING_PROPERTIES` | RW | Returns a list of all properties that are used by OpenVINO cache to build the hash key. | `N/A` | `N/A` |
| `ov::streams::num`/</br>`NUM_STREAMS` | RO | Not used by the NPU plugin.</br> Always set to 1. | `AUTO/`</br>`INT` | `1` |
| `ov::optimal_number_of_infer_requests`/</br>`OPTIMAL_NUMBER_OF_INFER_REQUESTS` | RO | Returns the optimal number of inference requests to be used by the application. Depends on the platform version and on ov::hint::performance_mode. Please see the table below. | `N/A` | `N/A` |
| `ov::range_for_async_infer_requests`/</br>`RANGE_FOR_ASYNC_INFER_REQUESTS` | RO | Returns a tuple (bottom, top, step). </br> Not used by the NPU plugin. | `N/A` | `N/A` |
| `ov::range_for_streams`/</br>`RANGE_FOR_STREAMS` | RO | Returns a tuple (bottom, top).</br> Not used by the NPU plugin. | `N/A`| `N/A` |
| `ov::enable_profiling`/</br>`PERF_COUNT` | RW | Enables or disables performance counters. | `YES`/ `NO` | `NO` |
| `ov::hint::performance_mode`/</br>`PERFORMANCE_HINT` | RW | Sets the performance profile used to determine default values of DPUs/DMAs/NIREQs.</br>Default values for each profile are documented below. | `THROUGHPUT`/</br>`LATENCY`/</br>`UNDEFINED` | `UNDEFINED` |
| `ov::hint::num_requests`/</br>`PERFORMANCE_HINT_NUM_REQUESTS` | RW | Sets the number of outstanding inference requests. | `[0-]` | `1` |
| `ov::hint::model_priority`/</br>`MODEL_PRIORITY` | RW | Assigns a priority for the model execution. | `LOW`/</br>`MEDIUM`/</br>`HIGH` | `MEDIUM` |
| `ov::hint::enable_cpu_pinning`/</br>`ENABLE_CPU_PINNING` | RW | Allows CPU threads pinning during inference. | `YES`/ `NO` /</br>`NO` 
| `ov::log::level`/</br>`LOG_LEVEL` | RW |  Sets the log level for NPU Plugin. An environment variable is also made available to expose logs from early initialization phase: OV_NPU_LOG_LEVEL. | `LOG_LEVEL_NONE`/</br>`LOG_LEVEL_ERROR`/</br>`LOG_LEVEL_WARNING`/</br>`LOG_LEVEL_DEBUG`/</br>`LOG_LEVEL_TRACE` |  `_NONE` |
| `ov::cache_dir`/</br>`CACHE_DIR` | RW | Folder path to be used by the OpenVINO cache. | `N/A` | empty |
| `ov::available_devices`/</br>`AVAILABLE_DEVICES` | RO | Returns the list of enumerated NPU devices. </br> NPU plugin does not currently support multiple devices. | `N/A`| `N/A` |
| `ov::device::id`/</br>`DEVICE_ID` | RW | Device identifier. Empty means auto detection. | empty/</br> `3700`/</br> `3720` | empty |
| `ov::device::uuid`/</br> | RO | Returns the Universal Unique ID of the NPU device. | `N/A`| `N/A` |
| `ov::device::architecture`/</br>`DEVICE_ARCHITECTURE` | RO | Returns the platform information. | `N/A`| `N/A` |
| `ov::device::full_name`/</br>`FULL_DEVICE_NAME` | RO | Returns the full name of the NPU device. | `N/A`| `N/A` |
| `ov::internal::exclusive_async_requests`/</br>`EXCLUSIVE_ASYNC_REQUESTS` | RW | Allows to use exclusive task executor for asynchronous infer requests. | `YES`/ `NO`| `NO` |
| `ov::intel_vpux::dpu_groups`/</br>`NPU_DPU_GROUPS` | RW | Sets the number of DPU groups that will be used to execute the model. | `[1-4]` for NPU 3700,</br> `[1-2]` for NPU 3720 | `-1` |
| `ov::intel_vpux::dma_engines`/</br>`NPU_DMA_ENGINES` | RW | Sets the number of DMA engines that will be used to execute the model. | `[1-2]` for NPU 3700,</br> `[1-2]` for NPU 3720 |  `-1` |
| `ov::intel_vpux::compilation_mode`/</br>`NPU_COMPILATION_MODE` | RW | Selects different compilation pipelines. | `ReferenceSW`/</br>`ReferenceHW`/</br>`DefaultHW`/</br>`ShaveCodeGen` | empty |
| `ov::intel_vpux::compilation_mode_params`/</br>`NPU_COMPILATION_MODE_PARAMS` | RW | Sets various parameters supported by the NPU compiler. |  `<params>` | empty  |
| `ov::intel_vpux::compiler_type`/</br>`NPU_COMPILER_TYPE` | RW | Selects the type of NPU compiler to be used for compilation of a network. </br> 'DRIVER' is the default value.</br> 'MLIR' is the default value only when DEVELOPER_BUILD=ON. | `MLIR`/</br>`DRIVER` | `DRIVER` |
| `ov::intel_vpux::print_profiling`/</br>`NPU_PRINT_PROFILING` | RW | `NONE` - Do not print profiling info;</br>`TEXT`, `JSON` - Print detailed profiling info during inference in the requested format. | `NONE`/</br>`TEXT`/</br>`JSON` | `NONE` |
| `ov::intel_vpux::profiling_output_file`/</br>`NPU_PROFILING_OUTPUT_FILE` | RW | std::cout is used if parameter value was empty. | `<path>` | empty |
| `ov::intel_vpux::vpux_platform`/</br>`NPU_PLATFORM` | RW | Used to compile and run on a specific device.</br>If device is not available, creating infer request will throw an exception. | `AUTO_DETECT`/</br>`VPU3700`/</br>`VPU3720` | `AUTO_DETECT`|
| `ov::intel_vpux::use_elf_compiler_backend`/</br>`NPU_USE_ELF_COMPILER_BACKEND` | RW | Sets the format in which the compiled model is stored. | `YES`/ `NO`| `NO` |
| `ov::intel_vpux::device_total_mem_size`/</br>`NPU_DEVICE_TOTAL_MEM_SIZE` | RO | Returns the total device available memory size. |  `N/A`| `N/A` |
| `ov::intel_vpux::driver_version`/</br>`NPU_DRIVER_VERSION` | RO | Returns the driver version. | `N/A`| `N/A` |

Note: 'intel_vpux' namespace will be renamed and 'vpux' prefix will be removed in future releases.

&nbsp;
### Performance Hint: Default Number of DPU Groups / DMA Engines

The following table shows the default values for the number of DPU Groups (Tiles) and DMA Engines selected by the plugin based on the performance mode (THROUGHPUT/LATENCY) and based on the platform:

| Performance hint | NPU Platform        | Number of DPU Groups | Number of DMA Engines           |
| :---             | :---                | :---                 | :---                            |
| THROUGHPUT       | 3700                | 1                    | 1                               |
| THROUGHPUT       | 3720                | 2 (all of them)      | 2 (all of them)                 |
| LATENCY          | 3700                | 4 (all of them)      | 1                               |
| LATENCY          | 3720                | 2 (all of them)      | 2 (all of them)                 |

&nbsp;
### Performance Hint: Optimal Number of Inference Requests

The following table shows the optimal number of inference requests returned by the plugin based on the performance mode (THROUGHPUT/LATENCY) and based on the platform:

| NPU Platform        | Nr. of Inference Requests </br> THROUGHPUT  | Nr. of Inference Requests </br> LATENCY |
| :---                | :---                                        | :---                                    |
| 3700                | 8                                           | 1                                       |
| 3720                | 4                                           | 1                                       |


&nbsp;
## Stateful models

Key ingredients to support stateful models which distinguish them from other models are:
* Implementing ReadValue and Assign operators
* Implementing Query State API (to give user an API to reset/get/set states)
* Implementing initialization for a state

More details on how OpenVINO supports stateful models can be found here: [Stateful models](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_network_state_intro.html).

The current implementation of state variables inside the NPU plugin is illustrated by the below diagram:

![High Level Design](./img/stateful_models.png)

Notes on the implementation:
* One network with N inputs + K state variables + M outputs will be converted by the compiler into a model with (N+K) inputs and (M+K) outputs. State variables are represented by a set of input/output nodes. This is currently needed because the underlying software stack (driver and runtime) does not support state variables. 
* The input and output nodes corresponding to state variables have different buffers allocated through the Level Zero API.
* The content of the output buffer is copied back into the input buffer by the plugin through the use of an intermediate state buffer:
    * NPU Plugin allocates and maintains one additional state buffer which is exposed through the GetState/SetState API
    * The actual level zero input buffer for the state is updated when the inference is triggered with the content of the state buffer
    * The state buffer is updated once the inference is completed with the content of the output level zero buffer

The implementation of state variables in the NPU plugin will be improved for upcoming releases.

&nbsp;
## Dynamic shapes
Dynamic shapes are not supported by the NPU plugin yet.
