//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-sw-layers-to-VPUIP-UPA --canonicalize %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
func.func @SingleLayer(%arg0: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
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

// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: memref<20x1x128xf16>, %arg1: memref<20x1xf16>) -> memref<1x20x1x1xf16> {
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
func.func @ReduceL1(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %2 = VPU.ReduceL1(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "L1"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %2 = VPU.ReduceL2(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "L2"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func.func @ReduceProd(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %2 = VPU.ReduceProd(%0, %1) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x1xf16> to memref<1x32x112x1xf16>
    return %3 : memref<1x32x112x1xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {keep_dims, type = "PROD"}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<1xsi32>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func.func @Bucketize(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x112xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    %2 = VPU.Bucketize(%0, %1) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x112x112xsi32> to memref<1x32x112x112xsi32>
    return %3 : memref<1x32x112x112xsi32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.BucketizeUPA {output_type = si32, with_right_bound}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<2xsi32>) ouputs([[VAR0]] : memref<1x32x112x112xsi32>) -> memref<1x32x112x112xsi32>
    // CHECK:       return [[VAR1]] : memref<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func.func @ExtractImagePatches(%arg0: memref<64x3x10x10xf32>) -> memref<64x27x2x2xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<64x3x10x10xf32> to tensor<64x3x10x10xf32>
    %1 = VPU.ExtractImagePatches(%0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = #IE.pad_type<VALID>} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<64x27x2x2xf32> to memref<64x27x2x2xf32>
    return %2 : memref<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<64x27x2x2xf32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ExtractImagePatchesUPA {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:      inputs(%arg0 : memref<64x3x10x10xf32>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<64x27x2x2xf32>) -> memref<64x27x2x2xf32>
    // CHECK:       return [[VAR1]] : memref<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @Selu
func.func @Selu(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16> {
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

// CHECK-LABEL: @Roll
func.func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    %0 = VPU.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare memref<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    // CHECK-DAG:       [[VAR1:%.*]] = const.Declare memref<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x10x100x200xf16> to memref<3x10x100x200xf16>
    // CHECK:       [[VAR4:%.*]] = memref.alloc() : memref<3x10x100x200xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.RollUPA inputs([[VAR2]] : memref<3x10x100x200xf16>, [[VAR1]] : memref<1xsi32>, [[VAR0]] : memref<2xsi32>) outputs([[VAR4]] : memref<3x10x100x200xf16>) -> memref<3x10x100x200xf16>
    // CHECK:       [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<3x10x100x200xf16> to tensor<3x10x100x200xf16>
    // CHECK:       return [[VAR6]] : tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func.func @AdaptiveAvgPool(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x56x56xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %2 = VPU.AdaptiveAvgPool(%0, %1) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x32x56x56xf16> to memref<1x32x56x56xf16>
    return %3 : memref<1x32x56x56xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.AdaptiveAvgPoolUPA
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, [[CST]] : memref<2xsi32>) outputs([[VAR0]] : memref<1x32x56x56xf16>) -> memref<1x32x56x56xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func.func @AdaptiveMaxPool(%arg0: memref<1x32x112x112xf16>) -> (memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %2, %3 = VPU.AdaptiveMaxPool(%0, %1) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    %4 = builtin.unrealized_conversion_cast %2 : tensor<1x32x56x56xf16> to memref<1x32x56x56xf16>
    %5 = builtin.unrealized_conversion_cast %3 : tensor<1x32x56x56xsi32> to memref<1x32x56x56xsi32>
    return %4, %5 : memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x56x56xsi32>
    // CHECK:       [[VAR2:%.*]], [[VAR3:%.*]] = VPUIP.AdaptiveMaxPoolUPA {index_element_type = si32}
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>, %cst : memref<2xsi32>) outputs(%0 : memref<1x32x56x56xf16>, %1 : memref<1x32x56x56xsi32>) -> memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>
    // CHECK:       return [[VAR2]], [[VAR3]] : memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>
}

// -----

// CHECK-LABEL: @GatherND
func.func @GatherND(%arg0: memref<2x3x4xsi32>, %arg1: memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<2x3x4xsi32> to tensor<2x3x4xsi32>
    %cst = const.Declare memref<2x1x1xsi32> = dense<0> : tensor<2x1x1xsi32>
    %1 = memref.alloc() : memref<2x1x3x4xsi32>
    %2 = VPUIP.GatherNDUPA {batch_dims = 0 : i64} inputs(%arg0 : memref<2x3x4xsi32>, %cst : memref<2x1x1xsi32>) outputs(%1 : memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32>
    %3 = builtin.unrealized_conversion_cast %2 : memref<2x1x3x4xsi32> to tensor<2x1x3x4xsi32>
    %4 = builtin.unrealized_conversion_cast %3 : tensor<2x1x3x4xsi32> to memref<2x1x3x4xsi32>
    %5 = VPUIP.Copy inputs(%4 : memref<2x1x3x4xsi32>) outputs(%arg1 : memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32>
    return %5 : memref<2x1x3x4xsi32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<2x1x1xsi32> = dense<0> : tensor<2x1x1xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<2x1x3x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.GatherNDUPA {batch_dims = 0 : i64}
    // CHECK-SAME:      inputs(%arg0 : memref<2x3x4xsi32>, [[CST]] : memref<2x1x1xsi32>) outputs([[VAR0]] : memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[VAR1]] : memref<2x1x3x4xsi32>) outputs(%arg1 : memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32>
    // CHECK:       return [[VAR2]] : memref<2x1x3x4xsi32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
func.func @NonMaxSuppression(%arg0: memref<3x100x4xf16>, %arg1: memref<3x5x100xf16>, %arg2: memref<300x3xsi32>, %arg3: memref<300x3xf16>, %arg4: memref<1xsi32>) -> (memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x100x4xf16> to tensor<3x100x4xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<3x5x100xf16> to tensor<3x5x100xf16>
    %2, %3, %4 = VPU.NonMaxSuppression(%0, %1) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    %5 = builtin.unrealized_conversion_cast %2 : tensor<300x3xsi32> to memref<300x3xsi32>
    %6 = builtin.unrealized_conversion_cast %3 : tensor<300x3xf16> to memref<300x3xf16>
    %7 = builtin.unrealized_conversion_cast %4 : tensor<1xsi32> to memref<1xsi32>
    return %5, %6, %7 : memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<300x3xsi32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<300x3xf16>
    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1xsi32>
    // CHECK:       [[VAR3:%.*]], [[VAR4:%.*]], [[VAR5:%.*]]  = VPUIP.NonMaxSuppressionUPA {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64}
    // CHECK-SAME:      inputs(%arg0 : memref<3x100x4xf16>, %arg1 : memref<3x5x100xf16> outputs(%0 : memref<300x3xsi32>, %1 : memref<300x3xf16>, %2 : memref<1xsi32>) -> memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>
    // CHECK:       return [[VAR3]], [[VAR4]], [[VAR5]] : memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSum
    func.func @EmbeddingBagOffsetsSum(%arg0: memref<5x6x4xsi32>, %arg1: memref<4x6x4xsi32>) -> memref<4x6x4xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<5x6x4xsi32> to tensor<5x6x4xsi32>
    %1 = VPU.EmbeddingBagOffsetsSum(%0) {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 0, 2, 2], weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<4x6x4xsi32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<4x6x4xsi32> to memref<4x6x4xsi32>
    %3 = VPUIP.Copy inputs(%2 : memref<4x6x4xsi32>) outputs(%arg1 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    return %3 : memref<4x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<4x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingBagOffsetsSumUPA {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 0, 2, 2], weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} inputs(%arg0 : memref<5x6x4xsi32>) outputs(%0 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs(%1 : memref<4x6x4xsi32>) outputs(%arg1 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    // CHECK:       return [[VAR2]] : memref<4x6x4xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingSegmentsSum
func.func @EmbeddingSegmentsSum(%arg0: memref<5x6x4xsi32>, %arg1: memref<7x6x4xsi32>) -> memref<7x6x4xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<5x6x4xsi32> to tensor<5x6x4xsi32>
    %1 = VPU.EmbeddingSegmentsSum(%0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<7x6x4xsi32> to memref<7x6x4xsi32>
    return %2 : memref<7x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<7x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingSegmentsSumUPA {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} inputs(%arg0 : memref<5x6x4xsi32>) ouputs([[VAR0]] : memref<7x6x4xsi32>) -> memref<7x6x4xsi32>
    // CHECK:       return [[VAR1]] : memref<7x6x4xsi32>
}

// -----

// CHECK-LABEL: @NormalizeIE
 func.func @NormalizeIE(%arg0: memref<1x128x50x85xf16>, %arg1: memref<1x1x1x1xf16>) -> memref<1x128x50x85xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x128x50x85xf16> to tensor<1x128x50x85xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<1x1x1x1xf16> to tensor<1x1x1x1xf16>
    %2 = VPU.NormalizeIE(%0, %1) {across_spatial = true, channel_shared = true, eps = 1.000000e-05 : f64} : tensor<1x128x50x85xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x50x85xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x128x50x85xf16> to memref<1x128x50x85xf16>
    return %3 : memref<1x128x50x85xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x128x50x85xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.NormalizeIEUPA {across_spatial = true, channel_shared = true, eps = 1.000000e-05 : f64} inputs(%arg0 : memref<1x128x50x85xf16>, %arg1 : memref<1x1x1x1xf16>) outputs(%0 : memref<1x128x50x85xf16>) -> memref<1x128x50x85xf16>
    // CHECK:       return [[VAR1]] : memref<1x128x50x85xf16>
}

// -----

// CHECK-LABEL: @CumSum
func.func @CumSum(%arg0: memref<1x9xf16>) -> memref<1x9xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x9xf16> to tensor<1x9xf16>
    %1 = VPU.CumSum(%0) {axis_value = 1 : i64 , exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x9xf16> to memref<1x9xf16>
    return %2 : memref<1x9xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x9xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CumSumUPA {axis_value = 1 : i64, exclusive, reverse} inputs(%arg0 : memref<1x9xf16>) outputs(%0 : memref<1x9xf16>) -> memref<1x9xf16>
    // CHECK:       return [[VAR1]] : memref<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
 func.func @DeformablePSROIPooling(%arg0: memref<1x441x8x8xf16>, %arg1: memref<30x5xf16>) -> memref<30x49x3x3xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x441x8x8xf16> to tensor<1x441x8x8xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<30x5xf16> to tensor<30x5xf16>
    %2 = VPU.DeformablePSROIPooling(%0, %1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} : tensor<1x441x8x8xf16>, tensor<30x5xf16> -> tensor<30x49x3x3xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<30x49x3x3xf16> to memref<30x49x3x3xf16>
    return %3 : memref<30x49x3x3xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<30x49x3x3xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.DeformablePSROIPoolingUPA {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} inputs(%arg0 : memref<1x441x8x8xf16>, %arg1 : memref<30x5xf16>) outputs(%0 : memref<30x49x3x3xf16>) -> memref<30x49x3x3xf16>
    // CHECK:       return [[VAR1]] : memref<30x49x3x3xf16>
}

// -----

// CHECK-LABEL: @Tan
func.func @Tan(%arg0: memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x112x112xf16> to tensor<1x32x112x112xf16>
    %1 = VPU.Tan(%0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x32x112x112xf16> to memref<1x32x112x112xf16>
    return %2 : memref<1x32x112x112xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.TanUPA
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16>
    // CHECK:       return [[VAR1]] : memref<1x32x112x112xf16>
}
