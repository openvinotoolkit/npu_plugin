# VPUX Plugin

## Introducing VPUX Plugin

VPUX Plugin was developed for inference of neural networks on the supported Intel&reg; Movidius&trade; VPU devices:

  * Gen 3 Intel&reg; Movidius&trade; VPU (3700VE)
  * Gen 3 Intel&reg; Movidius&trade; VPU (3400VE)
  * Intel&reg; Movidius&trade; S 3900V VPU
  * Intel&reg; Movidius&trade; S 3800V VPU
  * Intel&reg; Vision Accelerator Design PCIe card with Intel Movidius&trade; S VPU

## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host              | OS (64-bit)                          |
| :---              | :---                                 |
| Development       | Ubuntu* 18.04, MS Windows* 10        |
| Target            | Ubuntu* 18.04, MS Windows* 10, Yocto |

### Offline Compilation

To run inference using VPUX plugin, Inference Engine Intermediate Representation needs to be compiled for a certain VPU device. Sometimes, compilation may take a while (several minutes), so it makes sense to compile a network before execution. Compilation can be done by a tool called `compile_tool`. An example of the command line running `compile_tool`:
```
compile_tool -d VPUX -m model.xml -c vpu.config
```
Where `VPUX` is a name of the plugin to be used, `model.xml` - a model to be compiled, `vpu.config` is a text file with config options. `vpu.config` must contain setting of `VPUX_PLATFORM` config option to define a VPU platform to be used for compilation. An example of creation a config file to compile a model for Gen 3 Intel&reg; Movidius&trade; VPU (3700VE):
```
echo "VPUX_PLATFORM VPU3700" > ./vpu.config
```

If the platform is not specified, you will get an error `Error: VPUXPlatform is not defined`.

The table below contains VPU devices and corresponding `VPUX_PLATFORM`:

| VPU device                                    | VPUX_PLATFORM |
| :-------------------------------------------  | :----------- |
| Gen 3 Intel&reg; Movidius&trade; VPU (3700VE) |   VPU3700    |
| Gen 3 Intel&reg; Movidius&trade; VPU (3400VE) |   VPU3400    |
| Intel&reg; Movidius&trade; S 3900V VPU        |   VPU3900    |
| Intel&reg; Movidius&trade; S 3800V VPU        |   VPU3800    |

## Supported Configuration Parameters


The VPUX plugin accepts the following options:

| Parameter Name        | Parameter Values | Default Value    | Description                                                                      |
| :---                  | :---             | :---             | :---                                                                             |
| `LOG_LEVEL`                  |`LOG_LEVEL_NONE`/ `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_DEBUG`/ `LOG_LEVEL_TRACE`|`LOG_LEVEL_NONE`  |Set log level for VPUX plugin |
| `PERF_COUNT`                 | `YES`/`NO`                                                                                   |`NO`              |Enable or disable performance counter|
| `DEVICE_ID`                  | `VPU-0`/ `VPU-1`/ `VPU-2`/ `VPU-3`                                                           | `VPU-0`          |Device identifier |
| `VPUX_PLATFORM`              | `VPU3400`/ `VPU3700`/ `VPU3800`/ `VPU3900`                                                   | `VPU3700`        |Device platform |
| `VPUX_THROUGHPUT_STREAMS`    | positive integer                                                                             | `2`              |Number of threads for model execution|
| `VPUX_INFERENCE_SHAVES`      | positive integer, `0`                                                                        | `0`              |Number of shaves for model execution, if `0` is set, count of SHAVEs will be evaluated automatically|
| `VPUX_CSRAM_SIZE`            | integer                                                                                      | `-1`             |Set the size of CSRAM in bytes, if `-1` is set, compiler will evaluate size of CSRAM automatically|
| `VPU_COMPILER_LOG_LEVEL`     | `LOG_LEVEL_NONE`/ `LOG_LEVEL_ERROR`/ `LOG_LEVEL_WARNING`/ `LOG_LEVEL_INFO`/ `LOG_LEVEL_TRACE`| `LOG_LEVEL_INFO` | Set log level for mcmCompiler |
| `VPU_COMPILER_CUSTOM_LAYERS` | string                                                                                       | empty            | Path to custom layer binding xml file. Custom layer has higher priority over native implementation |


# See Also

* [Inference Engine introduction](https://gitlab-icv.inn.intel.com/inference-engine/dldt/blob/master/docs/IE_DG/inference_engine_intro.md)
