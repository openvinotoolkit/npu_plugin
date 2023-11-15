//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @SingleLayer
func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: @LogSoftmax
func.func @LogSoftmax(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.LogSoftmax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = VPU.LogSoftmax(%arg0) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: @LSTMCell
func.func @LSTMCell(%arg0: tensor<1x512xf16>, %arg1: tensor<1x256xf16>, %arg2: tensor<1x256xf16>, %arg3: tensor<1024x512xf16>, %arg4: tensor<1024x256xf16>, %arg5: tensor<1024xf16>) -> (tensor<1x256xf16>, tensor<1x256xf16>) {
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
func.func @Broadcast(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16> {
    %cst = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.Broadcast(%arg0, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    return %0 : tensor<1x64x1x1xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR0:%.+]] = VPU.Broadcast(%arg0, [[CST]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>}
    // CHECK-SAME:    : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<1x64x1x1xf16>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func.func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = IE.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = #IE.pad_type<VALID>} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.+]] = VPU.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:    : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    // CHECK:       return [[VAR0]] : tensor<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated}
    // CHECK-SAME:    : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<8x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL1
func.func @ReduceL1(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func.func @ReduceProd(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func.func @Bucketize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xsi32> {
    %cst = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    %0 = IE.Bucketize(%arg0, %cst) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    return %0 : tensor<1x32x112x112xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.Bucketize(%arg0, [[CST]]) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @Selu
func.func @Selu(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = IE.Selu(%arg0) {alphaValue = 1.000000e+00 : f64, lambdaValue = 2.000000e+00 : f64, operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Selu(%arg0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @Roll
func.func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK-DAG: [[VAR0:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK-DAG: [[VAR1:%.+]] = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    // CHECK: [[VAR2:%.+]] = VPU.Roll(%arg0, [[VAR0]], [[VAR1]]) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    // CHECK: return [[VAR2]] : tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func.func @AdaptiveAvgPool(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0 = IE.AdaptiveAvgPool(%arg0, %cst) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %0 : tensor<1x32x56x56xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.AdaptiveAvgPool(%arg0, [[CST]]) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func.func @AdaptiveMaxPool(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>) {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0, %1 = IE.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    return %0, %1 : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.AdaptiveMaxPool(%arg0, [[CST]]) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
}

// -----

// CHECK-LABEL: @GatherND
func.func @GatherND(%arg0: tensor<5x7x3xsi32>) -> tensor<5x3xsi32> {
    %cst = const.Declare tensor<5x1xsi32> = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    %0 = IE.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    return %0 : tensor<5x3xsi32>

    // CHECK-DAG:               [[CST:%.+]] = const.Declare tensor<5x1xsi32>
    // CHECK-SAME{LITERAL}      = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    // CHECK:               [[VAR0:%.+]] = VPU.GatherND(%arg0, [[CST]]) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    // CHECK:               return [[VAR0]] : tensor<5x3xsi32>
}

// -----

// CHECK-LABEL: @GatherTree
func.func @GatherTree(%arg0: tensor<5x7x3xsi32>, %arg1: tensor<5x7x3xsi32>, %arg2: tensor<7xsi32>, %arg3: tensor<1xsi32>) -> tensor<5x7x3xsi32> {
    %0 = IE.GatherTree(%arg0, %arg1, %arg2, %arg3) : tensor<5x7x3xsi32>, tensor<5x7x3xsi32>, tensor<7xsi32>, tensor<1xsi32> -> tensor<5x7x3xsi32>
    return %0 : tensor<5x7x3xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.GatherTree(%arg0, %arg1, %arg2, %arg3) : tensor<5x7x3xsi32>, tensor<5x7x3xsi32>, tensor<7xsi32>, tensor<1xsi32> -> tensor<5x7x3xsi32>
    // CHECK: return [[VAR0]] : tensor<5x7x3xsi32>
}

// -----

// CHECK-LABEL: @GridSample
func.func @GridSample(%arg0: tensor<1x1x2x3xf16>, %arg1: tensor<1x1x3x2xf16>) -> tensor<1x1x1x3xf16> {
    %2 = IE.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    return %2 :  tensor<1x1x1x3xf16>

    // CHECK: [[VAR0:%.+]] = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x3xf16>
}

// -----

// CHECK-LABEL: @NormalizeL2
func.func @NormalizeL2(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[-1, 1]> : tensor<2xsi32>
    %0 = IE.NormalizeL2(%arg0, %cst) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<2xsi32> -> tensor<1x128x50x85xf16>
    return %0 : tensor<1x128x50x85xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<[-1, 1]> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.NormalizeL2(%arg0, [[CST]]) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<2xsi32> -> tensor<1x128x50x85xf16>
    // CHECK: return [[VAR0]] : tensor<1x128x50x85xf16>
}

// -----

// CHECK-LABEL: @GRUCell
func.func @GRUCell(%arg0: tensor<2x3xf16>, %arg1: tensor<2x4xf16>) -> tensor<2x4xf16> {
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
    // CHECK: [[VAR5:%.+]], [[VAR6:%.+]] = VPU.GRUSequence([[VAR0]], [[VAR1]], [[VAR2]], [[VAR3]], [[VAR4]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x3xf16>, tensor<2x1x4xf16>, tensor<1x12x3xf16>, tensor<1x12x4xf16>, tensor<1x12xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: [[VAR7:%.+]] = VPU.Reshape([[VAR6]]) {shape_value = [2, 4]} : tensor<2x1x4xf16> -> tensor<2x4xf16>
    // CHECK: return [[VAR7]] : tensor<2x4xf16>
}

// -----

// CHECK-LABEL: @GRUSequence
func.func @GRUSequence(%arg0: tensor<2x1x10xf16>, %arg1: tensor<2x1x4xf16>) -> (tensor<2x1x1x4xf16>, tensor<2x1x4xf16>) {
    %cst = const.Declare tensor<1x16xf16> = dense<1.0> : tensor<1x16xf16>
    %cst_0 = const.Declare tensor<1x12x4xf16> = dense<1.0> : tensor<1x12x4xf16>
    %cst_1 = const.Declare tensor<1x12x10xf16> = dense<1.0> : tensor<1x12x10xf16>
    %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    return %middle_hidden_state, %output_hidden_state : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>

    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
}

// -----

// CHECK-LABEL: @EmbeddingBagPackedSumWithWeights
func.func @EmbeddingBagPackedSumWithWeights(%arg0: tensor<5x10xf16>) -> tensor<3x10xf16> {
    %cst = const.Declare tensor<3x2xf16> = dense<9.997550e-02> : tensor<3x2xf16>
    %cst_0 = const.Declare tensor<3x2xsi32> = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    %0 = VPU.EmbeddingBagPackedSum(%arg0, %cst_0, %cst) : tensor<5x10xf16>, tensor<3x2xsi32>, tensor<3x2xf16> -> tensor<3x10xf16>
    return %0 : tensor<3x10xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3x2xf16> = dense<9.997550e-02> : tensor<3x2xf16>
    // CHECK: [[CST0:%.+]] = const.Declare tensor<3x2xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagPackedSum(%arg0, [[CST0]], [[CST]]) : tensor<5x10xf16>, tensor<3x2xsi32>, tensor<3x2xf16> -> tensor<3x10xf16>
    // CHECK: return [[VAR0]] : tensor<3x10xf16>
}

// -----

// CHECK-LABEL: @EmbeddingBagPackedSumNoWeights
func.func @EmbeddingBagPackedSumNoWeights(%arg0: tensor<5x10xf16>) -> tensor<3x10xf16> {
    %cst = const.Declare tensor<3x2xsi32> = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    %0 = IE.EmbeddingBagPackedSum(%arg0, %cst) : tensor<5x10xf16>, tensor<3x2xsi32> -> tensor<3x10xf16>
    return %0 : tensor<3x10xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3x2xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    // CHECK: [[CST0:%.+]] = const.Declare tensor<3x2xf16> = dense<1.000000e+00> : tensor<3x2xf16>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagPackedSum(%arg0, [[CST]], [[CST0]]) : tensor<5x10xf16>, tensor<3x2xsi32>, tensor<3x2xf16> -> tensor<3x10xf16>
    // CHECK: return [[VAR0]] : tensor<3x10xf16>
}

// -----

// CHECK-LABEL: @CumSum
func.func @CumSum(%arg0: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    %0 = IE.CumSum(%arg0, %cst) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16>, tensor<si32> -> tensor<1x9xf16>
    return %0 : tensor<1x9xf16>

    // CHECK: [[VAR0:%.+]] = VPU.CumSum(%arg0) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    // CHECK: return [[VAR0]] : tensor<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
  func.func @DeformablePSROIPooling(%arg0: tensor<1x441x8x8xf32>, %arg1: tensor<30x5xf32>) -> tensor<30x49x3x3xf32> {
    %0 = IE.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    return %0 : tensor<30x49x3x3xf32>

    // CHECK: [[VAR0:%.+]] = VPU.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    // CHECK: return [[VAR0]] : tensor<30x49x3x3xf32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
func.func @NonMaxSuppression(%arg0: tensor<3x100x4xf16>, %arg1: tensor<3x5x100xf16>) -> (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %0, %1, %2 = IE.NonMaxSuppression(%arg0, %arg1) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0]> : vector<6xi32>, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %0, %1, %2 : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] = VPU.NonMaxSuppression(%arg0, %arg1) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    // CHECK: return [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @OneHot
func.func @OneHot(%arg0: tensor<4xsi32>) -> tensor<4x3xf16> {
    %0 = IE.OneHot(%arg0) {axis_attr = 1 : i64, depth_attr = 3 : i64, off_value_attr = -1.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, outElemType = f16} : tensor<4xsi32> -> tensor<4x3xf16>
    return %0 : tensor<4x3xf16>

    // CHECK: [[VAR0:%.+]] = VPU.OneHot(%arg0) {axis = 1 : i64, depth = 3 : i64, off_value = -1.000000e+00 : f64, on_value = 1.000000e+00 : f64, outElemType = f16} : tensor<4xsi32> -> tensor<4x3xf16>
    // CHECK: return [[VAR0]] : tensor<4x3xf16>
}

// -----

// CHECK-LABEL: @ScatterElementsUpdate
func.func @ScatterElementsUpdate(%arg0: tensor<2x3x4xf16>, %arg1: tensor<1x3x1xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    %0 = IE.ScatterElementsUpdate(%arg0, %cst, %arg1) {axis_value = 1 : i64} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ScatterElementsUpdate(%arg0, %cst, %arg1) {axis = 1 : i64} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16> -> tensor<2x3x4xf16>
    // CHECK: return [[VAR0]] : tensor<2x3x4xf16>
}

// -----

// CHECK-LABEL: @Tan
func.func @Tan(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x112x112xf16>) {
    %0 = IE.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @ShapeCast
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ShapeCast(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16> {
    %0 = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs(%arg0 : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>

    // CHECK: [[VPU_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK: return [[VPU_SHAPE_CAST]]
}

// -----

// CHECK-LABEL: @DFT
// CHECK-SAME:   (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @DFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
    %0 = IE.DFT(%arg0) {axes_attr = [0, 1], operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    return %0 : tensor<10x4x2xf32>

    // CHECK: %0 = VPU.DFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    // CHECK: return %0 : tensor<10x4x2xf32>
}

// -----

// CHECK-LABEL: @IDFT
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @IDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
    %0 = IE.IDFT(%arg0) {axes_attr = [0, 1], operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    return %0 : tensor<10x4x2xf32>

    // CHECK: %0 = VPU.IDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    // CHECK: return %0 : tensor<10x4x2xf32>
}

// -----

// CHECK-LABEL: @RDFT
// CHECK-SAME:  (%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32>
func.func @RDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32> {
    %0 = IE.RDFT(%arg0) {axes_attr = [0, 1], operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x3x2x2xf32>
    return %0 : tensor<10x3x2x2xf32>

    // CHECK: %0 = VPU.RDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x3x2x2xf32>
    // CHECK: return %0 : tensor<10x3x2x2xf32>
}

// -----

// CHECK-LABEL: @IRDFT
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
    %0 = IE.IRDFT(%arg0) {axes_attr = [0, 1], operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
    return %0 : tensor<10x6xf32>

    // CHECK: %0 = VPU.IRDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
    // CHECK: return %0 : tensor<10x6xf32>
}

// -----

// CHECK-LABEL: @IRDFTOneAxis
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFTOneAxis(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
  %0 = IE.IRDFT(%arg0) {axes_attr = [1], operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, signal_size_attr = [-1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  return %0 : tensor<10x6xf32>

  // CHECK: %0 = VPU.IRDFT(%arg0) {axes_attr = [1], signal_size_attr = [-1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  // CHECK: return %0 : tensor<10x6xf32>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>

// CHECK: !qElemType0 = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>

// CHECK-LABEL: @InterpolateQuantized
func.func @InterpolateQuantized(%arg0: tensor<1x16x3x3x!qElemType0>) -> tensor<1x16x6x6x!qElemType1> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <NEAREST>,
                               shape_calc_mode = <SCALES>,
                               coord_mode = <ASYMMETRIC>,
                               nearest_mode = <FLOOR>,
                               antialias = false,
                               pads_begin = [0, 0, 0, 0],
                               pads_end = [0, 0, 0, 0],
                               cube_coeff = -7.500000e-01 : f64>,
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        axes_attr = [2, 3],
        scales_attr = [2.0, 2.0],
        sizes_attr = [6, 6]}
        : tensor<1x16x3x3x!qElemType0> -> tensor<1x16x6x6x!qElemType1>
    return %0 : tensor<1x16x6x6x!qElemType1>

    // CHECK:       [[VAL0:%.+]] = VPU.Interpolate(%arg0) {
    // CHECK-SAME:    attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:                           shape_calc_mode = <SCALES>,
    // CHECK-SAME:                           coord_mode = <ASYMMETRIC>,
    // CHECK-SAME:                           nearest_mode = <FLOOR>,
    // CHECK-SAME:                           antialias = false,
    // CHECK-SAME:                           pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:                           pads_end = [0, 0, 0, 0],
    // CHECK-SAME:                           cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:    axes_attr = [2, 3],
    // CHECK-SAME:    operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME:    scales_attr = [2.000000e+00, 2.000000e+00],
    // CHECK-SAME:    sizes_attr = [6, 6]}
    // CHECK-SAME:  : tensor<1x16x3x3x!qElemType0> -> tensor<1x16x6x6x!qElemType1>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @TopKWithKValue
// CHECK-SAME: (%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32>
func.func @TopKWithKValue(%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32> {
    %output_values, %target_shape = IE.TopK(%arg0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x64x128x128xf32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    return %target_shape : tensor<1x1x128x128xsi32>

    // CHECK: [[VALUES:%.*]], [[SHAPE:%.*]] = VPU.TopK(%arg0)
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
    // CHECK-SAME:         : tensor<1x64x128x128xf32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    // CHECK: return [[SHAPE]] : tensor<1x1x128x128xsi32>
}

// -----

// CHECK-LABEL: @TopKWithKConst
// CHECK-SAME: (%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32>
func.func @TopKWithKConst(%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32> {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x64x128x128xf32>, tensor<si32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    return %target_shape : tensor<1x1x128x128xsi32>

    // CHECK-DAG:   [[CST_K:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK: [[VALUES:%.*]], [[SHAPE:%.*]] = VPU.TopK(%arg0, [[CST_K]])
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
    // CHECK-SAME:         : tensor<1x64x128x128xf32>, tensor<si32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    // CHECK: return [[SHAPE]] : tensor<1x1x128x128xsi32>
}
