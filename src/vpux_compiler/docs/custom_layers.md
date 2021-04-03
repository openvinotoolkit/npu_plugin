# Custom Layers

Custom layers (like OCL) can be represented on **IE Dialect** as a call to an external Function.
This function will hold all requirements (precisions, layouts, memory spaces, etc.).
During the lowering to **VPUIP Dialect** this Function call will be replaced by the corresponding custom layer task.

## Custom Binary Layer

The **Custom Binary Layer** represents pre-compiled custom layer (eg. user-provided) on all Dialects.
The pre-compiled code in binary from is provided as separate constant operand.
The Operation for such kind of Layer holds the type of the layer (for example, `OCL`, `CPP`) to be used in lowering passes.
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
