# Architecture

The **VPUX NN Compiler** consists of the following parts:

* **Core utilities**.
* **FrontEnd**.
* **IE Dialect**.
* **IERT Dialect**.
* **VPUX/MCM Dialect**.
* **VPUIP Dialect**.
* **BackEnd**.
* **Conversion passes**.

## Core Utilities

The **VPUX NN Compiler** core utilities includes various auxiliary classes and functions to simplify IR interpretation and transformations:

* `src/experimental/vpux_compiler/include/vpux/compiler/utils/`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/ops_interfaces.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/static_allocation.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/const_content.hpp`

One part of the core utilities is tensor shape/stride/layout manipulation API:

* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dim.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dim_values.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dims_order.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/shape.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/strides.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/stride_reqs.hpp`

### Tensor Shape/Stride/Layout

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
These classes don't have implicit casting to integer, only explicit getter method for dimension index.
These classes are used as keys to access corresponding shape and strides arrays instead of plain integers.
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
* `Compact` - the stride for this dimension must not introduce gaps between neighbor elements in this dimension.
* `Aligned` - the byte stride for this dimension must be aligned by particular value.
* `Fixed` - the stride for this dimension must be equal to fixed value.

## FrontEnd

**FrontEnd** is used to import external source into MLIR infrastructure.
It supports the following sources:

* InferenceEngine `CNNNetwork` object - imported as **IE Dialect**.
* **TBD:** RunTime graph blob - imported as **VPUIP Dialect**.

The **FrontEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing, for example (see below).

## IE Dialect

The **IE Dialect** represents InferenceEngine/nGraph IR in terms of MLIR framework.

It works with MLIR Tensor Types only and performs HW-agnostic transformations.
It also provides common Interfaces and Validation methods for NN layers.

The network topology (nGraph) is represented as a MLIR Function, which works with `tensor` types.

```MLIR
func @main(%input: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %output = IE.SoftMax(%input) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %output
}
```

The network inputs and outputs information (names, precision, layout) is held in separate Operation - `IE.CNNNetwork`.

```MLIR
IE.CNNNetwork {
    entryPoint = @main, netName = "DynShapeExample"
} inputsInfo {
    IE.DataInfo {name = "input", precision = f32, layout = "NCHW">}
} outputsInfo {
    IE.DataInfo {name = "output", precision = f32, layout = "NC">}
}
```

## IERT Dialect

The **IERT Dialect** represents bufferized version of **IE Dialect**.

It introduces memory related details and scheduling logic into the IR.
It includes transformations and optimizations closer to HW level (memory re-usage, parallel resources usage, etc.).

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

## VPUIP Dialect

The **VPUIP Dialect** represents runtime graph schema.

It allows to work with the graph schema inside MLIR framework:

* Validate it.
* Perform additional low level transformations/optimizations.

## BackEnd

**BackEnd** is used to export **VPUX NN Compiler** IR into external output.
It supports the following modes:

* **VPUIP Dialect** serialization to runtime blob format.

The **BackEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing, for example (see below).

## Conversion Passes

These types of Passes performs lowering from high-level Dialects to low level.

## Custom Layers / eDSL

Custom layers (like OCL) can be represented on **IE Dialect** as a call to an external Function.
This function will hold all requirements (precisions, layouts, memory spaces, etc.).
During the lowering to **VPUIP Dialect** this Function call will be replaced by the corresponding custom layer task.

The eDSL based layers can also be represented as calls of separate Function, but the Function itself will be a part of common IR.
The only difference is that this Function will use another Dialect internally (for example, `Linalg` or `Affine`).
Such Functions can be transformed/optimized by a separate set of Passes during the common compilation pipeline.
This mechanism can also be used in the future for InferenceEngine layers optimization, when the layer is lowered to eDSL representation
and then is optimized using its compilation pipeline.
At the end, during lowering to **VPUIP Dialect** this Function call will also be replaced by appropriate tasks.

### Custom Binary Layer

The **Custom Binary Layer** represents pre-compiled custom layer (eg. user-provided) on all Dialects.
The pre-compiled code in binary from is provided as separate constant operand.
The Operation for such kind of Layer holds the type of the layer (for example, `OCL`, `CPP`, `eDSL`) to be used in lowering passes.
Also it holds various attributes to handle layouts/precision/strides requirements in generic passes.
In addition to the above attributes extra specific attributes can be attached to hold lower Dialect specific information.

```MLIR
#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>
#OYXI = affine_map<(o, i, y, x) -> (o, y, x, i)>

%bin_code = IE.Constant tensor<10240xui8> = dense<...>

%output = IE.CustomBinaryLayer VPUIP.OCL from %bin_code
    (%input : tensor<1x16x100x100xf16>, %weights : tensor<32x16x3x3xf16>)
    -> tensor<1x32x100x100xf16>
    {
        input_layouts = [#NHWC, #OYXI],
        output_layouts = [#NHWC],
        VPUIP.max_num_shaves = 8
    }
```

In the lowering pass from **IE Dialect** to **IERT Dialect** the `tensor` types will be bufferized to `memref`,
allocation will be added and layouts restriction will be used to adjust other operations.
Same passes will be used for allocation and reordering optimizations as for known Layer Operations.

```MLIR
#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>
#OYXI = affine_map<(o, i, y, x) -> (o, y, x, i)>

%bin_code = IERT.Constant memref<10240xui8> = dense<...>

%input_NHWC = alloc : memref<1x16x100x100xf16, #NHWC>
IERT.Reorder(%input, %input_NHWC)
dealloc %input

%output_NHWC = alloc : memref<1x32x100x100xf16, #NHWC>
IERT.CustomBinaryLayer VPUIP.OCL from %bin_code
    (%input_NHWC, %weights_OYXI, %output_NHWC)
    {
        VPUIP.max_num_shaves = 8
    }
dealloc %input_NHWC

%output = alloc : memref<1x32x100x100xf16>
IERT.Reorder(%output_NHWC, %output)
dealloc %output_NHWC
```

At the end the `IERT.CustomBinaryLayer` Operation will be lowered to corresponding Operation from **VPUIP Dialect** based on its type.

### Custom Layer

The **Custom Layer** in its turn holds the kernel code in the IR inside its Region.
It can use various MLIR Dialects for its representation (`linalg`, `tile`, `gpu`, etc.).
The Region is isolated from above and gets its operands from parent Operation only.

```MLIR
%conv = IE.CustomLayer VPUIP.CPP
    (%input as %arg0 : tensor<1x16x100x100xf16>, %weights as %arg1 : tensor<32x16x3x3xf16>)
    -> tensor<1x32x100x100xf16>
    {
        %0 = linalg.conv2d(%arg0, %arg1) { <attributes> }
        return %0
    }

%relu = IE.CustomLayer VPUIP.CPP
    (%conv as %arg0 : tensor<1x32x100x100xf16>)
    -> tensor<1x32x100x100xf16>
    {
        %0 = linalg.ReLU(%arg0)
        return %0
    }
```

Having such representation we can start the kernel implementation optimization on **IE Dialect** level.
First, we can try to merge sub-graph of the **Custom Layers** with the same type into single **Custom Layer**.

```MLIR
%relu = IE.CustomLayer VPUIP.CPP
    (%input as %arg0, %weights as %arg1)
    -> tensor<1x32x100x100xf16>
    {
        %0 = linalg.conv2d(%arg0, %arg1) { <attributes> }
        %1 = linalg.ReLU(%0)
        return %1
    }
```

The internal Region of the **Custom Layer** can be optimized with its own pipeline.

During translation to **IERT Dialect** bufferization take a place.
All allocation Operation from **Custom Layer** inner Region will be moved outside of it.

```MLIR
%input_NHWC = alloc : memref<1x16x100x100xf16, #NHWC>
%relu_NHWC = alloc : memref<1x32x100x100xf16, #NHWC>
%relu = alloc : memref<1x32x100x100xf16>

IERT.CustomLayer VPUIP.CPP
    (%input as %arg0, %weights_OYXI as %arg1, %input_NHWC as %arg2, %relu_NHWC as %arg3, %relu as %arg4)
    {
        linalg.transpose(%arg0, %arg2)
        linalg.conv2d_NHWC(%arg2, %arg3)
        linalg.ReLU(%arg3, %arg3)
        linalg.transpose(%arg3, %arg4)
        return
    }

dealloc %input
dealloc %input_NHWC
dealloc %relu_NHWC
```

After all optimizations for inner Region are applied and the Region body is lowered to some Dialect (`LLVM` or C-like),
the kernel code can be compiled into final machine code (either with separate tool or with LLVM chain).
Having compiled kernel code the **Custom Layer** can be replaced with **Custom Binary Layer**.
Then the **Custom Binary Layer** will be lowered to VPUIP Dialect (see above section)

## Dynamic Shapes Support

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
