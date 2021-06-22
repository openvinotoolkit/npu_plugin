# How to set devices and platforms

For setting platform for compilation and device for inference it has to use common config parameter `DEVICE_ID`. Config parameter `VPUX_PLATFORM` is currently deprecated. Parameter `DEVICE_ID` has the following possible formats:

| Format                                 | Compilation platform | Inference device |
| :-------------------------------------------  | :--------------- | :----------------- |
| empty |   Auto detection   | Auto detection |
| `platform` |   As specified    | Auto detection (for specified platform) |
| `platform.slice`|  As specified    | As specified |

## Platform for compilation

The table below contains VPU devices and corresponding VPU platform

| VPU device                                    | VPU platform |
| :-------------------------------------------  | :----------- |
| Gen 3 Intel&reg; Movidius&trade; VPU (3700VE) |   3700    |
| Gen 3 Intel&reg; Movidius&trade; VPU (3400VE) |   3400    |
| Intel&reg; Movidius&trade; S 3900V VPU        |   3900    |
| Intel&reg; Movidius&trade; S 3800V VPU        |   3800    |

Here are the examples:
```
compile_tool -d VPUX.3700 -m model.xml -c vpu.config
```
Compilation for Gen 3 Intel&reg; Movidius&trade; VPU (3700VE)

If the platform is not specified, VPUX Plugin tries to determine it by analyzing all available system devices:
```
compile_tool -d VPUX -m model.xml
```

If system doesn't have any devices and platform for compilation is not provided, you will get an error `No devices found - DEVICE_ID with platform is required for compilation`

## Device for inference

Here are the examples:
```
benchmark_app -d VPUX -m model.xml
```
Run inference on any available VPU device
```
benchmark_app -d VPUX.3900 -m model.xml
```
Run inference on any available slice of Intel&reg; Movidius&trade; S 3900V VPU
```
benchmark_app -d VPUX.3800.0 -m model.xml
```
Run inference on the first slice of Intel&reg; Movidius&trade; S 3800V VPU
