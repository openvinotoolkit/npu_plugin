# KMB Plugin

## Introducing KMB Plugin for Inference Engine

KMB Plugin is developed for inference of neural networks on new Gen3 Intel&reg; Movidius&trade; VPU Code-Named Keem Bay.

## Supported networks

Currently, KMB Plugin supports only INT8 models quantized using [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html) delivered with the Intel&reg; Distribution of OpenVIN&trade; toolkit release package.


## Offline compilation

NN Compilation is not available on ARM. Offline compilation step is required to be done using compile-tool from OpenVINO&trade; package for Ubuntu&reg;. 
Please refer to release notes for details. 
A pre-compiled model (blob) should be loaded to VPU via ImportNetwork API.


## Supported Configuration Parameters


The KMB plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| `VPU_COMPILER_LOG_LEVEL`    | `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_INFO`/ `LOG_LEVEL_TRACE`/ `LOG_LEVEL_NONE` | `LOG_LEVEL_INFO` | Set log level for mcmCompiler |
| `VPU_COMPILER_ELTWISE_SCALES_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable eltwise scales alignment |
| `VPU_COMPILER_CONCAT_SCALES_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable concat scales alignment |
| `VPU_COMPILER_WEIGHTS_ZERO_POINTS_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable weights zero points alignment |
| `KMB_THROUGHPUT_STREAMS`    | positive integer | 1 | Set the number of threads to use for model execution |
| `VPU_KMB_LOAD_NETWORK_AFTER_COMPILATION`    | `YES`/`NO` | `NO` | Enable or disable blob transfer to device if LoadNetwork is called |
| `VPU_COMPILER_COMPILATION_PASS_BAN_LIST` | std::string | empty | List of mcm passes to be removed from mcm compilation descriptor (value example: kmb_adapt,KMBQuantizeConversion;adapt,TileOps) |
| `VPU_COMPILER_CUSTOM_LAYERS` | std::string | empty | Path to custom layer binding xml file. Custom layer has higher priority over native implementation. |
| `VPU_KMB_COMPILATION_DESCRIPTOR_PATH`    | string | 'mcm_config/compilation' | Path to folder with compilation config files |
| `VPU_KMB_COMPILATION_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for network compilation |
| `VPU_KMB_TARGET_DESCRIPTOR_PATH`    | string | 'mcm_config/target' | Path to folder with target config files |
| `VPU_KMB_TARGET_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for target device |
| `VPU_COMPILER_USE_FUSE_SCALE_INPUT`    | `YES`/`NO` | `YES` | Enable or disable fusing scaleshift |


# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
