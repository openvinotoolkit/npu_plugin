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
    %cst = const.Declare tensor<4xsi32> = #const.Content<dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]>
    %0 = IE.Broadcast(%arg0, %cst) {mode = "BIDIRECTIONAL"} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    return %0 : tensor<1x64x1x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<4xsi32> = #const.Content<dense<1> : tensor<4xsi64>, [#const.ConvertElemType<si32>]>
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
    %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %0 = IE.ReduceL1(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK: [[VAR0:%.+]] = VPU.ReduceL1(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK: [[VAR0:%.+]] = VPU.ReduceL2(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func @ReduceProd(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %0 = IE.ReduceProd(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    // CHECK: [[VAR0:%.+]] = VPU.ReduceProd(%arg0, [[CST]]) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func @Bucketize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xsi32> {
    %cst = const.Declare tensor<2xsi32> = #const.Content<dense<[10, 20]> : tensor<2xsi32>>
    %0 = IE.Bucketize(%arg0, %cst) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    return %0 : tensor<1x32x112x112xsi32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = #const.Content<dense<[10, 20]> : tensor<2xsi32>>
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

// CHECK-LABEL: @AdaptiveAvgPool
func @AdaptiveAvgPool(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %cst = const.Declare tensor<2xsi32> = #const.Content<dense<[56, 56]> : tensor<2xsi32>>
    %0 = IE.AdaptiveAvgPool(%arg0, %cst) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %0 : tensor<1x32x56x56xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = #const.Content<dense<56> : tensor<2xsi32>>
    // CHECK: [[VAR0:%.+]] = VPU.AdaptiveAvgPool(%arg0, [[CST]]) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func @AdaptiveMaxPool(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>) {
    %cst = const.Declare tensor<2xsi32> = #const.Content<dense<[56, 56]> : tensor<2xsi32>>
    %0, %1 = IE.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    return %0, %1 : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2xsi32> = #const.Content<dense<56> : tensor<2xsi32>>
    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.AdaptiveMaxPool(%arg0, [[CST]]) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
}
