// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-layers-to-VPUIP --canonicalize %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
func @SingleLayer(%arg0: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>
    %1 = VPU.SoftMax(%0) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x1x1x1000xf16> to memref<1x1x1x1000xf16>
    return %2: memref<1x1x1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16>)
    // CHECK: return [[VAR1]] : memref<1x1x1x1000xf16>
}

// -----

// CHECK-LABEL: @ReshapeInGraph
func @ReshapeInGraph(%arg0: memref<1x256x2x1xf16>) -> memref<1x256x2x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x256x2x1xf16> to tensor<1x256x2x1xf16>
    %1 = VPU.Reshape(%0) {shape_value = [1, 512, 1, 1]} : tensor<1x256x2x1xf16> -> tensor<1x512x1x1xf16>
    %2 = VPU.SoftMax(%1) {axisInd = 1} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16, {mem_space = @DDR}>
    %3 = VPU.Reshape(%2) {shape_value = [1, 256, 2, 1]} : tensor<1x512x1x1xf16, {mem_space = @DDR}> -> tensor<1x256x2x1xf16, {mem_space = @DDR}>
    %4 = VPU.Copy(%3) : tensor<1x256x2x1xf16, {mem_space = @DDR}> -> tensor<1x256x2x1xf16>
    %5 = builtin.unrealized_conversion_cast %4 : tensor<1x256x2x1xf16> to memref<1x256x2x1xf16>
    return %5 : memref<1x256x2x1xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.GenericReshape inputs(%arg0 : memref<1x256x2x1xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x512x1x1xf16, @DDR>
    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, @DDR>)
    // CHECK:       [[VAR3:%.*]] = VPUIP.GenericReshape inputs([[VAR2]] : memref<1x512x1x1xf16, @DDR>) -> memref<1x256x2x1xf16, @DDR>
    // CHECK:       [[VAR4:%.*]] = memref.alloc() : memref<1x256x2x1xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x256x2x1xf16, @DDR>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x256x2x1xf16>) -> memref<1x256x2x1xf16>
    // CHECK:       return [[VAR5]] : memref<1x256x2x1xf16>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func @CTCGreedyDecoder(%arg0: memref<20x1x128xf16>, %arg1: memref<20x1xf16>) -> memref<1x20x1x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<20x1x128xf16> to tensor<20x1x128xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<20x1xf16> to tensor<20x1xf16>
    %2 = VPU.CTCGreedyDecoder(%0, %1) {mergeRepeated} : tensor<20x1x128xf16>, tensor<20x1xf16> -> tensor<1x20x1x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x20x1x1xf16> to memref<1x20x1x1xf16>
    return %3 : memref<1x20x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x20x1x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CTCGreedyDecoderUPA {mergeRepeated}
    // CHECK-SAME:      inputs(%arg0 : memref<20x1x128xf16>, %arg1 : memref<20x1xf16>) outputs([[VAR0]] : memref<1x20x1x1xf16>) -> memref<1x20x1x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL1
func @ReduceL1(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %2 = VPU.ReduceL1(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "L1"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func @ReduceL2(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %2 = VPU.ReduceL2(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "L2"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func @ReduceProd(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %2 = VPU.ReduceProd(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "PROD"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func @Bucketize(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x112xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = #const.Content<dense<[10, 20]> : tensor<2xsi32>>
    %2 = VPU.Bucketize(%0, %1) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x112xsi32> to memref<1x32x112x112xsi32>
    return %3 : memref<1x32x112x112xsi32>

    // CHECK:       [[CST:%.*]] = const.Declare memref<2xsi32> = #const.Content<dense<[10, 20]> : tensor<2xsi32>>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.BucketizeUPA {output_type = si32, with_right_bound}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<2xsi32>) ouputs([[VAR0]] : memref<1x32x112x112xsi32>) -> memref<1x32x112x112xsi32>
    // CHECK:       return [[VAR1]] : memref<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func @ExtractImagePatches(%arg0: memref<64x3x10x10xf32>) -> memref<64x27x2x2xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<64x3x10x10xf32> to tensor<64x3x10x10xf32>
    %1 = VPU.ExtractImagePatches(%0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = "VALID"} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<64x27x2x2xf32> to memref<64x27x2x2xf32>
    return %2 : memref<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<64x27x2x2xf32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ExtractImagePatchesUPA {autoPad = "VALID", rates = [1, 1], sizes = [3, 3], strides = [5, 5]} 
    // CHECK-SAME:      inputs(%arg0 : memref<64x3x10x10xf32>) 
    // CHECK-SAME:      outputs([[VAR0]] : memref<64x27x2x2xf32>) -> memref<64x27x2x2xf32>
    // CHECK:       return [[VAR1]] : memref<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @Selu
func @Selu(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = VPU.Selu(%0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x32x112x112xf16> to memref<1x32x112x112xf16>
    return %2 : memref<1x32x112x112xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.SeluUPA {alphaValue = 1.000000e+00 : f64, lambdaValue = 2.000000e+00 : f64}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func @AdaptiveAvgPool(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x56x56xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = #const.Content<dense<[56, 56]> : tensor<2xsi32>>
    %2 = VPU.AdaptiveAvgPool(%0, %1) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x56x56xf16> to memref<1x32x56x56xf16>
    return %3 : memref<1x32x56x56xf16>

    // CHECK:       [[CST:%.*]] = const.Declare memref<2xsi32> = #const.Content<dense<56> : tensor<2xsi32>>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.AdaptiveAvgPoolUPA
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<2xsi32>) outputs([[VAR0]] : memref<1x32x56x56xf16>) -> memref<1x32x56x56xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func @AdaptiveMaxPool(%arg0: memref<1x32x112x112xf16>) -> (memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = #const.Content<dense<[56, 56]> : tensor<2xsi32>>
    %2, %3 = VPU.AdaptiveMaxPool(%0, %1) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    %4 = builtin.unrealized_conversion_cast %2 : tensor<1x32x56x56xf16> to memref<1x32x56x56xf16>
    %5 = builtin.unrealized_conversion_cast %3 : tensor<1x32x56x56xsi32> to memref<1x32x56x56xsi32>
    return %4, %5 : memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>

    // CHECK:       [[CST:%.*]] = const.Declare memref<2xsi32> = #const.Content<dense<56> : tensor<2xsi32>>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x56x56xsi32>
    // CHECK:       [[VAR2:%.*]], [[VAR3:%.*]] = VPUIP.AdaptiveMaxPoolUPA {index_element_type = si32}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, %cst : memref<2xsi32>) outputs(%0 : memref<1x32x56x56xf16>, %1 : memref<1x32x56x56xsi32>) -> memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>
    // CHECK:       return [[VAR2]], [[VAR3]] : memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>
}
