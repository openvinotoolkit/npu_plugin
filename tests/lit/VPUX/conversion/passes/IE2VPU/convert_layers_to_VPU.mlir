//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @SingleLayer
func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: @LSTMCell
func @LSTMCell(%arg0: tensor<1x512xf16>, %arg1: tensor<1x256xf16>, %arg2: tensor<1x256xf16>, %arg3: tensor<1024x512xf16>, %arg4: tensor<1024x256xf16>, %arg5: tensor<1024xf16>) -> (tensor<1x256xf16>, tensor<1x256xf16>) {
    %hiddenState, %cellState = IE.LSTMCell(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {hiddenSize = 256}
        : tensor<1x512xf16>, tensor<1x256xf16>, tensor<1x256xf16>, tensor<1024x512xf16>, tensor<1024x256xf16>, tensor<1024xf16>
        -> tensor<1x256xf16>, tensor<1x256xf16>
    return %hiddenState, %cellState : tensor<1x256xf16>, tensor<1x256xf16>

    // CHECK:       [[VAL0:%.+]], [[VAL1:%.+]] = VPU.LSTMCell(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {hiddenSize = 256 : i64}
    // CHECK-SAME:    : tensor<1x512xf16>, tensor<1x256xf16>, tensor<1x256xf16>, tensor<1024x512xf16>, tensor<1024x256xf16>, tensor<1024xf16>
    // CHECK-SAME:    -> tensor<1x256xf16>, tensor<1x256xf16>
    // CHECK: return [[VAL0]], [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Broadcast
func @Broadcast(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16> {
    %cst = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.Broadcast(%arg0, %cst) {mode = "BIDIRECTIONAL"} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    return %0 : tensor<1x64x1x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR0:%.+]] = VPU.Broadcast(%arg0, [[CST]]) {mode = "BIDIRECTIONAL"}
    // CHECK-SAME:    : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<1x64x1x1xf16>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = IE.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = "VALID"} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.+]] = VPU.ExtractImagePatches(%arg0) {autoPad = "VALID", rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:    : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    // CHECK:       return [[VAR0]] : tensor<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated}
    // CHECK-SAME:    : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<8x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL1
func @ReduceL1(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ReduceL1(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR0:%.+]] = VPU.ReduceL1(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR0:%.+]] = VPU.ReduceL2(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func @ReduceProd(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ReduceProd(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR0:%.+]] = VPU.ReduceProd(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func @Bucketize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xsi32> {
    %cst = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    %0 = IE.Bucketize(%arg0, %cst) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    return %0 : tensor<1x32x112x112xsi32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.Bucketize(%arg0, [[CST]]) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @Selu
func @Selu(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = IE.Selu(%arg0) {alphaValue = 1.000000e+00 : f64, lambdaValue = 2.000000e+00 : f64, operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Selu(%arg0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @Roll
func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK: [[VAR0:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR1:%.+]] = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR2:%.+]] = VPU.Roll(%arg0, [[VAR0]], [[VAR1]]) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    // CHECK: return [[VAR2]] : tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func @AdaptiveAvgPool(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0 = IE.AdaptiveAvgPool(%arg0, %cst) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %0 : tensor<1x32x56x56xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.AdaptiveAvgPool(%arg0, [[CST]]) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func @AdaptiveMaxPool(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>) {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0, %1 = IE.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    return %0, %1 : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.AdaptiveMaxPool(%arg0, [[CST]]) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
}

// -----

// CHECK-LABEL: @GatherND
func @GatherND(%arg0: tensor<5x7x3xsi32>) -> tensor<5x3xsi32> {
    %cst = const.Declare tensor<5x1xsi32> = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    %0 = IE.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    return %0 : tensor<5x3xsi32>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<5x1xsi32>
    // CHECK-SAME{LITERAL}      = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    // CHECK:               [[VAR0:%.+]] = VPU.GatherND(%arg0, [[CST]]) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    // CHECK:               return [[VAR0]] : tensor<5x3xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSum
func @EmbeddingBagOffsetsSum(%arg0: tensor<5x6x4xsi32>) -> tensor<2x6x4xsi32> {
    %0 = IE.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>,
    weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<2x6x4xsi32>
    return %0 : tensor<2x6x4xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2],
    // CHECK-SAME: weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<2x6x4xsi32>
    // CHECK: return [[VAR0]] : tensor<2x6x4xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingSegmentsSum
func @EmbeddingSegmentsSum(%arg0: tensor<5x6x4xsi32>) -> tensor<7x6x4xsi32> {
    %0 = IE.EmbeddingSegmentsSum(%arg0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
        operand_segment_sizes = dense<[1, 0, 0, 0, 0, 0]> : vector<6xi32>, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01],
        segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    return %0 : tensor<7x6x4xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingSegmentsSum(%arg0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
    // CHECK-SAME:  per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    // CHECK: return [[VAR0]] : tensor<7x6x4xsi32>
}

// -----

// CHECK-LABEL: @GridSample
func @GridSample(%arg0: tensor<1x1x2x3xf16>, %arg1: tensor<1x1x3x2xf16>) -> tensor<1x1x1x3xf16> {
    %2 = IE.GridSample(%arg0, %arg1) {align_corners, mode = "BILINEAR", padding_mode = "BORDER"} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    return %2 :  tensor<1x1x1x3xf16>

    // CHECK: [[VAR0:%.+]] = VPU.GridSample(%arg0, %arg1) {align_corners, mode = "BILINEAR", padding_mode = "BORDER"} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x3xf16>
}

// -----

// CHECK-LABEL: @GRUCell
func @GRUCell(%arg0: tensor<2x3xf16>, %arg1: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<1.0> : tensor<12x3xf16>
    %cst_0 = const.Declare tensor<12x4xf16> = dense<1.0> : tensor<12x4xf16>
    %cst_1 = const.Declare tensor<12xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 9.390620e+00, 1.000980e+00, 2.152340e+00, 3.720700e+00, 9.992180e+00, 2.320310e+00, 3.125000e+00, 1.000000e+01]> : tensor<12xf16>
    %0 = IE.GRUCell(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, hidden_size = 4 : i64, should_linear_before_reset} : tensor<2x3xf16>, tensor<2x4xf16>, tensor<12x3xf16>, tensor<12x4xf16>, tensor<12xf16> -> tensor<2x4xf16>
    return %0 : tensor<2x4xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Reshape(%arg0) {shape_value = [2, 1, 3]} : tensor<2x3xf16> -> tensor<2x1x3xf16>
    // CHECK: [[VAR1:%.+]] = VPU.Reshape(%arg1) {shape_value = [2, 1, 4]} : tensor<2x4xf16> -> tensor<2x1x4xf16>
    // CHECK: [[VAR2:%.+]] = VPU.Reshape(%cst) {shape_value = [1, 12, 3]} : tensor<12x3xf16> -> tensor<1x12x3xf16>
    // CHECK: [[VAR3:%.+]] = VPU.Reshape(%cst_0) {shape_value = [1, 12, 4]} : tensor<12x4xf16> -> tensor<1x12x4xf16>
    // CHECK: [[VAR4:%.+]] = VPU.Reshape(%cst_1) {shape_value = [1, 12]} : tensor<12xf16> -> tensor<1x12xf16>
    // CHECK: [[VAR5:%.+]], [[VAR6:%.+]] = VPU.GRUSequence([[VAR0]], [[VAR1]], [[VAR2]], [[VAR3]], [[VAR4]]) {clip = 0.000000e+00 : f64, direction = "FORWARD", hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x3xf16>, tensor<2x1x4xf16>, tensor<1x12x3xf16>, tensor<1x12x4xf16>, tensor<1x12xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: [[VAR7:%.+]] = VPU.Reshape([[VAR6]]) {shape_value = [2, 4]} : tensor<2x1x4xf16> -> tensor<2x4xf16>
    // CHECK: return [[VAR7]] : tensor<2x4xf16>
}

// -----

// CHECK-LABEL: @GRUSequence
func @GRUSequence(%arg0: tensor<2x1x10xf16>, %arg1: tensor<2x1x4xf16>) -> (tensor<2x1x1x4xf16>, tensor<2x1x4xf16>) {
    %cst = const.Declare tensor<1x16xf16> = dense<1.0> : tensor<1x16xf16>
    %cst_0 = const.Declare tensor<1x12x4xf16> = dense<1.0> : tensor<1x12x4xf16>
    %cst_1 = const.Declare tensor<1x12x10xf16> = dense<1.0> : tensor<1x12x10xf16>
    %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = "FORWARD", hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    return %middle_hidden_state, %output_hidden_state : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>

    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = "FORWARD", hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
}

// -----

// CHECK-LABEL: @CumSum
func @CumSum(%arg0: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    %0 = IE.CumSum(%arg0, %cst) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16>, tensor<si32> -> tensor<1x9xf16>
    return %0 : tensor<1x9xf16>

    // CHECK: [[VAR0:%.+]] = VPU.CumSum(%arg0) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    // CHECK: return [[VAR0]] : tensor<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
  func @DeformablePSROIPooling(%arg0: tensor<1x441x8x8xf32>, %arg1: tensor<30x5xf32>) -> tensor<30x49x3x3xf32> {
    %0 = IE.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = "BILINEAR_DEFORMABLE", output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    return %0 : tensor<30x49x3x3xf32>

    // CHECK: [[VAR0:%.+]] = VPU.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = "BILINEAR_DEFORMABLE", output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    // CHECK: return [[VAR0]] : tensor<30x49x3x3xf32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
func @NonMaxSuppression(%arg0: tensor<3x100x4xf16>, %arg1: tensor<3x5x100xf16>) -> (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %0, %1, %2 = IE.NonMaxSuppression(%arg0, %arg1) {box_encoding = "CENTER", iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0]> : vector<6xi32>, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %0, %1, %2 : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] = VPU.NonMaxSuppression(%arg0, %arg1) {box_encoding = "CENTER", iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    // CHECK: return [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @Tan
func @Tan(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x112x112xf16>) {
    %0 = IE.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @ShapeCast
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func @ShapeCast(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16> {
    %0 = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs(%arg0 : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>

    // CHECK: [[VPU_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK: return [[VPU_SHAPE_CAST]]
}
