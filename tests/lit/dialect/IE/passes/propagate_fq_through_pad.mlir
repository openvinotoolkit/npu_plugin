// RUN: vpux-opt --split-input-file --propagate-fq-through-pad --canonicalize %s | FileCheck %s

func @PropagateFqThroughPad(%arg0: tensor<1x2x1x512xf16>) -> tensor<1x64x1x512xf16> {
    %IN_LO = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<1x1x1x1xf16>>
    %IN_HI = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<6.142580e-01> : tensor<1x1x1x1xf16>>
    %WGHT = const.Declare tensor<64x2x1x7xf16> = #const.Content<dense<1.0> : tensor<64x2x1x7xf16>>
    %PAD_CST = const.Declare tensor<1x2x1x6xf16> = #const.Content<dense<0.000000e+00> : tensor<1x2x1x6xf16>>
    %WGHT_LO = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<-1.270000e+02> : tensor<1x1x1x1xf16>>
    %WGHT_HI = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.270000e+02> : tensor<1x1x1x1xf16>>

    %FQ_IN = IE.FakeQuantize(%arg0, %IN_LO, %IN_HI, %IN_LO, %IN_HI) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    %PAD = IE.Concat(%FQ_IN, %PAD_CST) {
        per_axis = {axis = 3 : i64}
    } : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>

    %FQ_WGHT = IE.FakeQuantize(%WGHT, %WGHT_LO, %WGHT_HI, %WGHT_LO, %WGHT_HI) {
        auto_broadcast = "NUMPY",
        levels = 255 : i64
    } : tensor<64x2x1x7xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<64x2x1x7xf16>

    %CONV2D = IE.Convolution(%PAD, %FQ_WGHT) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x2x1x518xf16>, tensor<64x2x1x7xf16> -> tensor<1x64x1x512xf16>

    %FQ_OUT = IE.FakeQuantize(%CONV2D, %WGHT_LO, %WGHT_HI, %WGHT_LO, %WGHT_HI) {
        auto_broadcast = "NUMPY",
        levels = 255 : i64
    } : tensor<1x64x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x1x512xf16>

    return %FQ_OUT : tensor<1x64x1x512xf16>

    // CHECK: %[[WGHT_HI:.*]] = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.270000e+02> : tensor<1x1x1x1xf16>>
    // CHECK: %[[WGHT_LO:.*]] = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<-1.270000e+02> : tensor<1x1x1x1xf16>>
    // CHECK: %[[PAD_CST:.*]] = const.Declare tensor<1x2x1x6xf16> = #const.Content<dense<0.000000e+00> : tensor<1x2x1x6xf16>>
    // CHECK: %[[WGHT:.*]] = const.Declare tensor<64x2x1x7xf16> = #const.Content<dense<1.000000e+00> : tensor<64x2x1x7xf16>>
    // CHECK: %[[IN_HI:.*]] =  const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<6.142580e-01> : tensor<1x1x1x1xf16>>
    // CHECK: %[[IN_LO:.*]] = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<1x1x1x1xf16>>

    // CHECK: %[[FQ_IN:.*]] = IE.FakeQuantize(%arg0, %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = "NUMPY",
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    // CHECK: %[[FQ_PAD_CST:.*]] = IE.FakeQuantize(%[[PAD_CST]], %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = "NUMPY",
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x6xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x6xf16>

    // CHECK: %[[PAD:.*]] = IE.Concat(%[[FQ_IN]], %[[FQ_PAD_CST]]) {
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0, 0], [0, 0, 0, 512]]
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>

    // CHECK: %[[FQ_PAD:.*]] = IE.FakeQuantize(%[[PAD]], %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = "NUMPY",
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x518xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x518xf16>

    // CHECK: %[[FQ_WGHT:.*]] = IE.FakeQuantize(%[[WGHT]], %[[WGHT_LO]], %[[WGHT_HI]], %[[WGHT_LO]], %[[WGHT_HI]]) {
    // CHECK-SAME:    auto_broadcast = "NUMPY",
    // CHECK-SAME:    levels = 255 : i64
    // CHECK-SAME: } : tensor<64x2x1x7xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<64x2x1x7xf16>

    // CHECK: %[[CONV2D:.*]] = IE.Convolution(%[[FQ_PAD]], %[[FQ_WGHT]]) {
    // CHECK-SAME:    dilations = [1, 1],
    // CHECK-SAME:    pads_begin = [0, 0],
    // CHECK-SAME:    pads_end = [0, 0],
    // CHECK-SAME:    strides = [1, 1]
    // CHECK-SAME: } : tensor<1x2x1x518xf16>, tensor<64x2x1x7xf16> -> tensor<1x64x1x512xf16>

    // CHECK: %[[FQ_CONV2D:.*]] = IE.FakeQuantize(%[[CONV2D]], %[[WGHT_LO]], %[[WGHT_HI]], %[[WGHT_LO]], %[[WGHT_HI]]) {
    // CHECK-SAME:    auto_broadcast = "NUMPY",
    // CHECK-SAME:    levels = 255 : i64
    // CHECK-SAME: } : tensor<1x64x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x1x512xf16>

    // CHECK: return %[[FQ_CONV2D]] : tensor<1x64x1x512xf16>
}
