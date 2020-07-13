# KMB Plugin

## Introducing KMB Plugin

The Inference Engine KMB plugin is developed for inference of neural networks on Intel&reg; Movidius&trade; VPU Code-Named Keem Bay.

## Supported networks

The Inference Engine KMB plugin supports the following models:

**INT8 quantized:**

* Inception v1, v3
* MobileNet v2
* ResNet-50
* SqueezeNet v1.1
* YOLO Tiny v2
* YOLO v2

Currently, the KMD plugin supports only quantized models. To quantize the model, you can use the Post-Training Optimization Tool delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

## Supported Configuration Parameters

The KMB plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| `VPU_COMPILER_LOG_LEVEL`    | `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_INFO`/ `LOG_LEVEL_TRACE`/ `LOG_LEVEL_NONE` | `LOG_LEVEL_INFO` | Set log level for mcmCompiler |
| `VPU_COMPILER_ELTWISE_SCALES_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable eltwise scales alignment |
| `VPU_COMPILER_CONCAT_SCALES_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable concat scales alignment |
| `VPU_COMPILER_WEIGHTS_ZERO_POINTS_ALIGNMENT`    | `YES`/`NO` | `YES` | Enable or disable weights zero points alignment |
| `VPU_KMB_PLATFORM`    | `VPU_2490` | `VPU_2490` | Set the target device |
| `VPU_KMB_THROUGHPUT_STREAMS`    | positive integer | 1 | Set the umber of threads to use for model execution |
| `VPU_KMB_LOAD_NETWORK_AFTER_COMPILATION`    | `YES`/`NO` | `NO` | Enable or disable blob transfer to device if LoadNetwork is called |
| `VPU_KMB_COMPILATION_DESCRIPTOR_PATH`    | string | 'mcm_config/compilation' | Path to folder with compilation config files |
| `VPU_KMB_COMPILATION_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for network compilation |
| `VPU_KMB_TARGET_DESCRIPTOR_PATH`    | string | 'mcm_config/target' | Path to folder with target config files |
| `VPU_KMB_TARGET_DESCRIPTOR`    | string | 'release_kmb' | Name of config file for target device |

## See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
