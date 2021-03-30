# Dynamic Shapes Support

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
