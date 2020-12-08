# VPUX NN Compiler

[TOC]

## Introduction

The **VPUX NN Compiler** is a new experimental NN compiler for VPU platforms.
It is based on [MLIR framework](https://mlir.llvm.org/) and utilizes its API and features.

### Goals

The **VPUX NN Compiler** is designed to achive the following goals:

* Improve network coverage for current VPU generation.
* Make new network enablement process lighter and faster.
* Extend compilation features set:
  * Tensors with arbitrary number of dimensions.
  * Dynamic shapes.
  * Control flow.
* Improve compilation performance.
* Integrate existing solutions from different projects in single code base.
* Design future-proof architecture with extensibility to the next VPU generations.
* Improve developer experience (debuggability, self-validation techniques, testing approach).

### MLIR Framework

The **VPUX NN Compiler** utilizes the following feature from MLIR to improve developer experience:

* IR manipulations.
* Transformations and pass management.
* IR self-validation.
* Unit testing.
* Debugging.

More information about the MLIR framework can be found on the [wiki page](https://wiki.ith.intel.com/display/VPUWIKI/MLIR+Framework).

## Design Principles

The **VPUX NN Compiler** architecture and its implementation is based on the following principles:

1. Explicit notion for IR validity and invariants.
2. Enforced architectural stability and self-validation during compilation pipeline.
3. IR spliting onto separate stages with different level of details.
4. Operation interfaces for generic passes.
5. Atomic and pipeline passes.

For more details about the first two principles please refer to [separate chapter](architectural_stability.md).

The third principle is achivied by MLIR architecture - Dialects concept.
The **VPUX NN Compiler** consists of several Dialects with different level of details.
The IR is lowered from high level abstractions to more detailed representation step-by-step during compilation pipeline.

The forth principle encourages using such MLIR concepts as Operation Traits and Interfaces.
They allow to reduce code duplication and group similar Operations unser single API.
Operation Interfaces also allows to write more generic passes, which are not bound to particular operation set.

The fifth principle declarase that each Pass in compilation pipeline must represent one single transformation
to reach one particular goal (either IR adaptation or IR optimization).
Such "atomic" pass is easier to be covered by unit testing.
The "atomic" passes can be joined togerther in the compilation chain inside "pipeline" pass.
The "pipeline" pass doesn't perform IR transformation on its own, instead it creates internal
pipeline of other passes (either "atomic" or another "pipeline") using MLIR dynamic pass manager feature.
The goal of "pipeline" pass is to establish correct order of underlying passes, while keep actual transformation logic inside them.

## Architecture

The **VPUX NN Compiler** consists of the following parts:

* **Core utilities**.
* **FrontEnd**.
* **IE Dialect**.
* **IERT Dialect**.
* **VPUX/MCM Dialect**.
* **VPUIP Dialect**.
* **BackEnd**.
* **Conversion passes**.

### Core Utilities

The **VPUX NN Compiler** core utilities includes various auxiliary classes and functions to simplify IR interpretation and transformations.

One part of the core utilities is tensor shape/stride/layout manipulation API.

#### Tensor Shape/Stride/Layout

The **VPUX NN Compiler** uses the following terms for the Tensor dimensions:

* **Logical** dimensions.
* **Memory** dimensions.

Those terms (**logical** and **memory**) are also applied to tensor shape and strides.
For example, the **logical** shape means that the dimensions sizes are assigned to **logical** dimensions.

**Logical** dimensions are abstracted from actual memory buffer layout.
Their order tensor shape is fixed and matches InferenceEngine, nGraph and MLIR order.
The actual meaning of each **logical** dimension is a property concrete Operation.
For example, Convolution interprets **logical** shape of activations tensor as `[N, C, H, W]`
and **logical** shape of weights tensor as `[O, I, KY, KX]`.

**Memory** dimensions, in contrast, are bound to actual memory layout and ordered from minor (most inner) to major (most outer).
They are used to work with memory buffers in common efficient way.

Both **logical** and **memory** dimensions are represented as separate classes (which internally holds single integer value - dimension index).
The `Dim` class represents **logical** dimension, while `MemDim` represents **memory** dimension.
This classes doesn't have implicit casting to integer, only explicit getter method for dimension index.
This classes are used as keys to access corresponding shape and strides arrays instead of plain integers.
In the same way, shape has two implementations (`Shape` and `MemSpace`) and strides (`Strides` and `MemStrides`).
The usage of separate classes (while they have common implementation logic) allows to catch all misuse of those two abstractions at compile time.

The `DimsOrder` class represents memory layout information.
It holds permutation array (in packed format) from **logical** dimensions to **memory** dimensions.
This class provides API to convert between those two representations in both way.
The class also provides API to work with MLIR class (`AffineMap`), which represents more generic layout description.

The final utility class in this section is `StrideReqs`.
It is used to collect various requirements for strides from different places and to calculate the strides based on this information.
It supports the following requirements:

* `Any` - means that there is no special requirements for particular dimension.
* `Compact` - the stride for this dimension must no introduce gaps between neighbour elements in this dimension.
* `Aligned` - the byte stride for this dimension must be aligned by particular value.
* `Fixed` - the stride for this dimension must be equal to fixed value.

### FrontEnd

**FrontEnd** is used to import external source into MLIR infrastructure.
It supports the following sources:

* InferenceEngine `CNNNetwork` object - imported as **IE Dialect**.
* **TBD:** RunTime graph blob - imported as **VPUIP Dialect**.

The **FrontEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing, for example (see below).

### IE Dialect

The **IE Dialect** represents InferenceEngine/nGraph IR in terms of MLIR framework.

It works with MLIR Tensor Types only and performs HW-agnostic transformations.
It also provides common Interfaces and Validation methods for NN layers.

The network topology (nGraph) is represented as MLIR Function, which works with `tensor` types.

```MLIR
func @main(%input: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %output = IE.SoftMax(%input) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %output
}
```

The network inputs and outputs information (names, precision, layout) is hold in separate Operation - `IE.CNNNetwork`.

```MLIR
IE.CNNNetwork {
    entryPoint = @main, netName = "DynShapeExample"
} inputsInfo {
    IE.DataInfo {name = "input", precision = f32, layout = "NCHW">}
} outputsInfo {
    IE.DataInfo {name = "output", precision = f32, layout = "NC">}
}
```

### IERT Dialect

**TBD:** add initial implementation for this Dialect.

The **IERT Dialect** represents bufferized version of **IE Dialect**.

It brings memory related details into IR as well as initial scheduling logic.

```MLIR
#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>

func @main(%input: memref<1x3x240x240xf16, #NHWC>, %output: memref<1x3x240x240xf16, #NHWC>) {
    %temp = alloc : memref<1x3x240x240xf16, #NHWC, /*memory space*/ 1>
    IERT.SoftMax(%input, %temp) {axisInd = 1 : i32}
    linalg.copy(%temp, %output)
    dealloc %temp
    return
}
```

### VPUX/MCM Dialect

**TBD:** to be designed.

Those dialects handles VPU-specifics, such as:

* NCE operations.
* Memory hierarchy.
* Executors hierarchy (NCEs, SHAVEs, DMAs).
* Asynchronous scheduling between different executors.
* Quantization and sparsity requirements.

### VPUIP Dialect

The **VPUIP Dialect** represents runtime graph schema.

It allows to work with graph schema inside MLIR framework:

* Validate it.
* Perform additional low level transformations/optimizations.

### BackEnd

**BackEnd** is used to export **VPUX NN Compiler** IR into external output.
It supports the following modes:

* **VPUIP Dialect** serialization to runtime blob format.

The **BackEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing, for example (see below).

### Conversion Passes

This types of Passes performs lowering from high-level Dialects to low level.

### Custom Layers / eDSL

**TBD:** finalize design.

Custom layers (like OCL) can be represented on **IE Dialect** as a call of external Function.
The function will hold all requirements (precisions, layouts, memory spaces, etc.).
During the lowering to **VPUIP Dialect** those Function call will be replaced by corresponding custom layer task.

The eDSL based layers can also be represented as calls of separate Function, but the Function itself will be a part of common IR.
The only difference is that this Function will use another Dialects internally (for example, `Linalg` or `Affine`).
Such Functions can be transformer/optimized by separate set of Passes during the common compilation pipeline.
This mechanism can also be used in future for InferenceEngine layers optimization, when the layer is lowered to eDSL representation
and then is optimized using its compilation pipeline.
At the end, during lowering to **VPUIP Dialect** this Function call will also be replaced by appripriate tasks.

### Dynamic Shapes Support

**TBD:** finalize design.

The **VPUX NN Compiler** will support dynamic shapes during the whole compilation stack,
using different representation ways on different abstraction levels.

It will allow to use 2 modes for dynamic shapes processing:

1. The actual result shape is calculated inside some runtime kernel (**preferable**).
2. The actual result shape is calculated with separate code.

Initial IR in **IE Dialect** will use Tensor types with dynamic shapes as is:

```MLIR
// Input batch, height, width are dynamic; channels are static.
// Output is all dynamic.
// Rank is static.
func @main(%input: tensor<?x3x?x?xf16>) -> tensor<?x?xf16> {
    // This layer has some formula to calculate output shape from input and attributes.
    // The runtime implementation of the layer will calculate the output shape internally.
    %temp0 = IE.Layer1(%input) : tensor<?x3x?x?xf16> -> tensor<?x16x?x?xf16>

    // This is a simple element-wise layer, output shape is equal to input shape.
    %temp1 = IE.SimpleLayer(%temp0) : tensor<?x16x?x?xf16> -> tensor<?x16x?x?xf16>

    // This is a custom layer, it has provided metadata for output shape calaculation,
    // but it must be done separately.
    %temp2 = IE.Custom(%temp1) : tensor<?x16x?x?xf16> -> tensor<?x?xf16>

    return %2
}
```

Separate Pass will add explicit notion of shapes into the IR.
The shapes will be represented as separate Values (Operation arguments and results).
All Operations with dynamic inputs will get additional arguments with actual shape per each dynamic input.
Operation that calculates result shape internally (mode 1) will return it as additional result (implicit tuple).
Operation that uses separate code for result shape computation (mode 2) will get it as one more additional argument.
The shape computation code will be inserted prior to this Operation.

```MLIR
func @main(%input: tensor<?x3x?x?xf16>) -> tensor<?x?xf16>, !shape.type {
    // Extract actual shape for network input
    %input_shape = shape.shape_of %input : tensor<?x3x?x?xf16> -> !shape.shape

    // Layer1 takes the shape of its argument as additional parameter.
    // Layer1 will compute the output shape internally and it returns it as additional result (implicit tuple).
    %temp0:2 = IE.Layer1(%input)[%input_shape] : tensor<?x3x?x?xf16>, !shape.shape -> tensor<?x16x?x?xf16>, !shape.type

    // SimpleLayer takes the shape of its argument as additional parameter.
    // Since for SimpleLayer output shape == input shape, it doesn't add additonal value for it,
    // instead it notifies the Pass to reuse existing Shape Value.
    %temp1 = IE.SimpleLayer(%temp0#0)[%temp0#1] : tensor<?x16x?x?xf16>, !shape.shape -> tensor<?x16x?x?xf16>

    // %temp0#1 (the shape of %temp0 tensor) is used as shape value for %temp1.
    // Custom layer has separate code for output shape calculation, so it is called prior to the Custom Operation.
    %temp2_shape = call @calc_custom_shape(%temp0#1) : !shape.shape -> !shape.shape

    // Custom takes the shape of its argument as additional parameter.
    // Custom takes the shape of its result as additional parameter (since it is calculated separately).
    %temp2 = IE.Custom(%temp1#0)[%temp0#1][%temp2_shape] : tensor<?x16x?x?xf16>, !shape.shape -> tensor<?x?xf16>

    return %temp2, %temp2_shape
}
```

Next Pass will calculate upper bound for the dynamic shapes to allow static memory allocation.

```MLIR
func @main(%input: tensor<100x3x300x300xf16>) -> tensor<100x1000xf16>, !shape.type {
    %input_shape = IE.ActualShapeOf %input : tensor<100x3x300x300xf16> -> !shape.shape

    %temp0:2 = IE.Layer1(%input)[%input_shape] : tensor<100x3x300x300xf16>, !shape.shape -> tensor<100x16x200x200xf16>, !shape.type

    %temp1 = IE.SimpleLayer(%temp0#0)[%temp0#1] : tensor<100x16x200x200xf16>, !shape.shape -> tensor<100x16x200x200xf16>

    %temp2_shape = call @calc_custom_shape(%temp0#1) : !shape.shape -> !shape.shape

    %temp2 = IE.Custom(%temp1#0)[%temp0#1][%temp2_shape] : tensor<100x16x200x200xf16>, !shape.shape -> tensor<100x1000xf16>

    return %temp2, %temp2_shape
}
```

At the final stage the shape Values will be lowered (bufferized) into `MemRef` types in the same way as tensors.

```MLIR
func @main(%input: memref<100x3x300x300xf16>, %output:memref<100x1000xf16>, %output_shape:memref<2xui32>) {
    // Extract actual shape for network input.
    // The memory for this Shape Value is allocated by caller (InferenceManager).
    %input_shape = IERT.ActualShapeOf %input : memref<100x3x300x300xf16> -> memref<4xui32>

    // Allocate memory for Layer1 result
    %temp0 = alloc : memref<100x16x200x200xf16>

    // Allocate memory for Layer1 result shape
    %temp0_shape = alloc : memref<4xui32>

    // On this step Operation takes both inputs and outputs as arguments, since they are allocated outside.
    IERT.Layer1(%input, %output)[%input_shape, %temp0_shape]

    // Allocate memory for SimpleLayer result.
    %temp1 = alloc : memref<100x16x200x200xf16>

    IE.SimpleLayer(%temp0, %temp1)[%temp0_shape]

    dealloc %temp0

    call @calc_custom_shape(%temp0_shape, %output_shape)

    // Custom Operation will write directly to %output.
    IE.Custom(%temp1, %output)[%temp0_shape, %output_shape]

    dealloc %temp0_shape

    return
}
```

After that alloc/dealloc pairs can be replaced with statically allocated memory offsets, using DDR for Shapes.

## Build and Test Instructions

The **VPUX NN Compiler** is a part of **kmb-plugin** repository and is built as a part of common build.
It can be enabled in plugin via `IE_VPUX_USE_EXPERIMENTAL_COMPILER=1` environment variable.

### Unit Tests

The **VPUX NN Compiler** uses two kind of unit tests:

* **GoogleTest** based
* **LLVM LIT** based

#### GoogleTest based Unit Tests

The *GoogleTest* based unit tests for the **VPUX NN Compiler** is avaialble in `vpuxUnitTestsExperimental` application.
It can be used as plain *GoogleTest* based application (including all command line options) without any specific environment setup.

#### LLVM LIT based Unit Tests

The *LLVM LIT* based unit tests uses Python scripts to run pattern-match-like tests.
This tests requires Python 3.0 to be installed.

The tests are copied to OpenVINO binary directory (`<openvino>/bin/<arch>/<build-type>/lit-tests`).
To run them use the following command:

```bash
cd <openvino>/bin/<arch>/<build-type>/lit-tests
python3 lit-tool/lit.py -v VPUX
```

For native x86 build the LIT tests can also be run via CTest tool:

```bash
cd <kmb-plugin-build-dir>
ctest -VV -R LIT-VPUX
```

**Note:** In order to run the LIT tests on updated test sources the `make all` command (or its analogue) must be run prior
to copy the updated test sources into OpenVINO binary directory.

### Functional tests

Existing functional tests for KMB plugin (`kmbFuncTests`) can be used with the **VPUX NN Compiler**.
The `IE_VPUX_USE_EXPERIMENTAL_COMPILER=1` environment variable must be set prior to their execution for both x86 side and ARM side.
