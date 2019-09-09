# KeemBay Plugin

## Introducing KeemBay Plugin

The Inference Engine KeemBay plugin is developed for inference of neural networks on Intel&reg; Movidius&trade; KeemBay SoC and Intel&reg.

## Installation on Linux* OS

For installation instructions, refer to the [Installation Guide for Linux\*](./docs/install_guides/installing-openvino-linux.md).

## Supported networks

The Inference Engine KeemBay plugin supports the following networks:

**TensorFlow (INT8 quantized)\***:
* Inception v1, v3
* MobileNet v2
* ResNet-50
* YOLO Tiny v1
* YOLO v2

**ONNX (INT8 quantized)\***:
* SqueezeNet v1.1
* SSD512

## Supported Configuration Parameters

See VPU common configuration parameters for the [VPU Plugins](./docs/IE_DG/supported_plugins/VPU.md).

In addition to common parameters, the KeemBay plugin accepts the following options:

| Parameter Name                | Parameter Values                                   | Default       |Description|
| :---                          | :---                                               | :---          | :---      |
| `MCM_TARGET_DESCRIPTOR_PATH`     | path to target JSON for mcmCompiler                         | `""`        | If empty effective path will be `"config/target"`
| `MCM_TARGET_DESCRIPTOR`          | target description JSON filename for mcmCompiler            | `""`        | If empty effective name will be `"ma2490"`
| `MCM_COMPILATION_DESCRIPTOR_PATH` | path to compilation description JSON for mcmCompiler        | `""`        | If empty effective path will be `"config/compilation"`
| `MCM_COMPILATION_DESCRIPTOR`     | compilation description JSON filename for mcmCompiler        | `""`        | If empty effective name will be `"debug_ma2490"`
| `MCM_GENERATE_BLOB`                                                                            | `YES`/`NO` | `YES` |
| `MCM_PARSING_ONLY`                                                                             | `YES`/`NO` | `YES` |
| `MCM_GENERATE_JSON`                                                                            | `YES`/`NO` | `YES` |
| `MCM_GENERATE_DOT`                                                                             | `YES`/`NO` | `NO`  |
| `MCM_COMPILATION_RESULTS_PATH`                                                                  |`<path to compilation results>` | `""` | If empty effective path will be `"."`
| `MCM_COMPILATION_RESULTS`                                                                       |`<name of mcmCompilator resulting files (blob, json, dot and png)>`| `""` | If empty effective name will be `"<network name>"`
| `KMB_EXECUTOR`                                                                                 | `YES`/`NO` | `NO` |

## See Also

* [Supported Devices](./docs/IE_DG/supported_plugins/Supported_Devices.md)
* [VPU Plugins](./docs/IE_DG/supported_plugins/VPU.md)
