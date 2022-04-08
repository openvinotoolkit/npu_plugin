# VPUX Plugin

## Introduction

The VPUX Plugin is a plugin for Inference Engine which allows running inference on VPU platforms.

### High Level Design

The plugin was designed with intention to support different API for interacting with VPU under single entry point. It has module structure which consists of two kinds of modules: compilers and backends.

@startuml
!theme lightgray

component InferenceEngine
component VPUXPlugin
component VPUXAbstractionLayer
frame " " {
    component Compilers
    component EngineBackends
}

InferenceEngine --> VPUXPlugin
VPUXPlugin --> VPUXAbstractionLayer
VPUXAbstractionLayer <.. Compilers
VPUXAbstractionLayer <.. EngineBackends
@enduml


### VPUX Abstraction Layer
The plugin implementation is based on intermediate API called VPUX Abstraction Layer. The API is designed to abstract concrete implementation of VPU Compiler and VPUX runtime available for VPU devices.

The first key component of the plugin is a compiler. Compiler is responsible for transforming ngraph [TBD ngraph reference] representation  of a network into a format which can be executed by VPU device.

The diagram below displays relashionship among interfaces and classes related to compilation. Refer to the links below for more detailed description of interfaces:

* \ref vpux::ICompiler "ICompiler"
* \ref vpux::INetworkDescription "INetworkDescription"


@startuml
!theme lightgray
interface INetworkDescription
interface ICompiler

ICompiler .right.> INetworkDescription

ConcreteCompiler -up-> ICompiler
ConcreteNetworkDescription -up-> INetworkDescription

@enduml

The second key component of the plugin is an engine backend. Engine backend defines platform (x86, aarch64), operation system (Windows10, Ubuntu18, Yocto), VPU device and mode(offline compilation, inference) which are going to be used for execution.

The diagram below displays relashionshep among interface and classsed related to engine backends. Refer to the links below for more detailed description of interfaces.

* \ref vpux::IEngineBackend "IEngineBackend"
* \ref vpux::IDevice "IDevice"
* \ref vpux::Executor "Executor"
* \ref vpux::Allocator "Allocator"


TBD: names are inconsistent

@startuml
!theme lightgray
interface IEngineBackend
interface IDevice
interface IExecutor
interface IAllocator /' mark it optional '/

IEngineBackend *.down.> IDevice
IDevice .down.> IExecutor
IDevice *-down-> IAllocator

ConcreteEngineBackend -left-> IEngineBackend
ConcreteDevice        -left-> IDevice
ConcreteExecutor      -up-> IExecutor
ConcreteAllocator     -up-> IAllocator
@enduml

### Supported Compilers

* MCM Compiler
* MLIR Compiler

(TBD links)

### Supported Engine Backends

* VPUAL backend (Yocto AARCH64)
* HDDL2 backend (Ubuntu18 x86)
* L0 backend (Windows10 x86)
* Emulator backend (Ubuntu18 x86) - only for purposes of debugging networks

(TBD links)

## VPUX Plugin developer guide

### Loading compilers

### Loading engine backends

### Offline Compilation

To run inference using VPUX plugin, Inference Engine Intermediate Representation needs to be compiled for a certain VPU device. Sometimes, compilation may take a while (several minutes), so it makes sense to compile a network before execution. Compilation can be done by a tool called `compile_tool`. An example of the command line running `compile_tool`:
```
compile_tool -d VPUX.3700 -m model.xml
```
Where `VPUX` is a name of the plugin to be used, `model.xml` - a model to be compiled, `3700` defines a VPU platform to be used for compilation.

If the platform is not specified, you will get an error `Error: VPUXPlatform is not defined`.

The table below contains VPU devices and corresponding platform:

| VPU device                                    | PLATFORM |
| :-------------------------------------------  | :--------|
| Gen 3 Intel&reg; Movidius&trade; VPU (3700VE) |   3700   |
| Gen 3 Intel&reg; Movidius&trade; VPU (3400VE) |   3400   |
| Intel&reg; Movidius&trade; S 3900V VPU        |   3900   |
| Intel&reg; Movidius&trade; S 3800V VPU        |   3800   |

### Models caching

### Choosing device for inference

VPU devices available in system can be obtained using a sample `hello_query_device`. The output below is an example of output (information about metrics was omitted for readability) for for Intel&reg; Vision Accelerator Design PCIe card with Intel Movidius&trade; S VPU

```
user@user:~/openvino/bin/intel64/Release$ ./hello_query_device
Available devices:
	Device: VPUX.3800.0

    ...

	Device: VPUX.3800.1

    ...

	Device: VPUX.3900.0

    ...

	Device: VPUX.3900.1

    ...

	Device: VPUX.3900.2

    ...

	Device: VPUX.3900.3

    ...
```

Device name for VPUX plugin follows the following format `VPUX[.platform[.id]]`.
`[platform]` and `[id]` parts can be omited.
If `[id]` is omited, the plugin will choose the first available device for inference with given `[platform]`. For example, if user passes a string `VPUX.3800`, a device `VPUX.3800.0` will be chosen for inference.
If both `[id]` and `[platform]` are ommited, then behavior depends on a platform used for inference:

1. All devices available will be utilized for inference on Ubuntu18 for x86
2. The first available device will be utilized for inference on Windows10 for x86 and Yocto for AARCH64

### Inference details

#### Inputs and outputs allocation

#### Pre-processing of input

* Resize, precision and layout conversion

#### Inference

#### Post-processing of output

* Precision and layout conversion

#### Config options

### VPUX Abstraction Layer Developer Guide

#### Adding a compiler

VPUX Plugin has modular structure and one of its modules is a graph compiler. Responsibility of the graph compiler is to transform a model from ngraph representation to a format which can be executed by a VPU device. The graph compiler shall implement interfaces \ref vpux::ICompiler "ICompiler" and \ref vpux::INetworkDescription "INetworkDescription". Implementation of those interfaces allows VPUX Abstraction Layer to include the compiler into VPUX Plugin.

Refer to the links below for more detailed description of interfaces to be implemented by a concrete compiler:

* \ref vpux::ICompiler "ICompiler"
* \ref vpux::INetworkDescription "INetworkDescription"

The graph compiler module shall be implemented as a runtime loaded library with the only one function exported: `CreateVPUXCompiler`. This function is an entry point to the compiler library and responsible for instantiating concrete implementation of \ref vpux::ICompiler "ICompiler" interface.  A typical implementation of `CreateVPUXCompiler` may look like the following:

```
INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<CompilerImpl>();
}
```

where `INFERENCE_PLUGIN_API` macro exports the function and `CompilerImpl` is a concrete implementation of \ref vpux::ICompiler "ICompiler" interface.

#### Including a compiler into VPUX Plugin

The selection of a compiler is controlled by a config option `VPUX_COMPILER_TYPE`. Currently, it supports two options: `MLIR` and `MCM` with the default value set to `MCM`.
A newly added compiler shall be registered:

1. Range of values for `VPUX_COMPILER_TYPE` shall be extended:

   * `src/vpux_al/include/vpux_private_config.hpp` to add a new value for the config option.
   * `src/vpux_al/src/config/compiler.cpp` to parse the value.

2. Factory method \ref vpux::Compiler::create "vpux::Compiler::create" shall be updated to create the new compiler if a corresponding config option is selected.

#### Adding an engine backend

### Level Zero Backend Developer Guide

#### Memory allocation

#### Device management

#### Inference details

#### Specific config option

The VPUX plugin supports the following private metrics:

| Metric Name                         | Metric Type | Description                            |
| :---                                | :---        | :---                                   |
| `VPUX_METRIC_KEY(BACKEND_NAME)`     | std::string |  The name of used backend              |


### VPUAL Backend Developer Guide

### HDDL2 Backend Developer Guide

## Contacts

* VPUX plugin developers team

TBD responsibilities
