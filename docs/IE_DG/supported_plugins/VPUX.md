# VPUX Plugin

## Introducing VPUX Plugin for Inference Engine

VPUX Plugin is developed for inference of neural networks on new Gen3 Intel&reg; Movidius&trade; VPU Code-Named Keem Bay.

## Supported networks

Currently, VPUX Plugin supports only INT8 models quantized using [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html) delivered with the Intel&reg; Distribution of OpenVIN&trade; toolkit release package.


## Offline compilation

NN Compilation is not available on ARM. Offline compilation step is required to be done using compile-tool from OpenVINO&trade; package for Ubuntu&reg;. 
Please refer to release notes for details. 
A pre-compiled model (blob) should be loaded to VPU via ImportNetwork API.


## Supported Configuration Parameters


The VPUX plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| `LOG_LEVEL`    | `LOG_LEVEL_NONE`/ `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_DEBUG` / `LOG_LEVEL_TRACE` | `LOG_LEVEL_NONE` | Set log level for VPUX plugin |
| `PERF_COUNT` | `YES`/`NO` | `NO` | Enable or disable performance counter |
| `DEVICE_ID`    | `VPU-0`/ `VPU-1`/ `VPU-2`/ `VPU-3` | `VPU-0` | `VPU-0` | Device identifier |
| `VPUX_PLATFORM`    | `VPU3400_A0`/ `VPU3400`/ `VPU3700`/ `VPU3800`/ `VPU3900` | `VPU3720` | Device platform |
| `VPUX_THROUGHPUT_STREAMS`    | positive integer | 2 | Set the number of threads to use for model execution |
| `KMB_THROUGHPUT_STREAMS`    | positive integer | 2 | **[Deprecated]** Set the number of threads to use for model execution |
| `VPUX_INFERENCE_SHAVES`    | positive integer | 0 | Set the number of shaves to be used by NNCore plug-in during inference. 0 - use default value |
| `CSRAM_SIZE`  | integer | -1 | Set the size of CSRAM in bytes |
| `VPU_COMPILER_LOG_LEVEL`    | `LOG_LEVEL_NONE`/ `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_INFO`/ `LOG_LEVEL_TRACE` | `LOG_LEVEL_INFO` | Set log level for mcmCompiler |
| `VPU_COMPILER_CUSTOM_LAYERS` | std::string | empty | Path to custom layer binding xml file. Custom layer has higher priority over native implementation |
| `VPU_COMPILER_COMPILATION_DESCRIPTOR_PATH`    | string | 'mcm_config/compilation' | Path to folder with compilation config files |
| `VPU_COMPILER_COMPILATION_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for network compilation |
| `VPU_COMPILER_TARGET_DESCRIPTOR_PATH`    | string | 'mcm_config/target' | Path to folder with target config files |
| `VPU_COMPILER_TARGET_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for target device |


# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
