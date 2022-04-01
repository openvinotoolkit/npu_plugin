// RUN: vpux-opt --split-input-file --adjust-for-vpu="swap-concat-with-eltwise" %s | FileCheck %s

func @SwapConcatWithLayer(%arg0: tensor<1x64x8x512xf16>, %arg1: tensor<1x64x1x512xf16>) -> tensor<1x64x9x512xf16> {
    %CST_FQ_LO = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<1x1x1x1xf16>>
    %CST_FQ_HI = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.970210e-01> : tensor<1x1x1x1xf16>>
    %CONV_WEIGHTS = const.Declare tensor<64x64x1x1xf16> = #const.Content<dense<1.0> : tensor<64x64x1x1xf16>>
    %CONV = IE.Convolution(%arg0, %CONV_WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x64x8x512xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x8x512xf16>

    %CONCAT = IE.Concat(%CONV, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    } : tensor<1x64x8x512xf16>, tensor<1x64x1x512xf16> -> tensor<1x64x9x512xf16>

    %PRELU = IE.LeakyRelu(%CONCAT) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x64x9x512xf16> -> tensor<1x64x9x512xf16>

    %OUT_FQ = IE.FakeQuantize(%PRELU, %CST_FQ_LO, %CST_FQ_HI, %CST_FQ_LO, %CST_FQ_HI) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x64x9x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x9x512xf16>

    return %OUT_FQ : tensor<1x64x9x512xf16>

    // CHECK:   [[CST_FQ_LO:%.*]] = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<1x1x1x1xf16>>
    // CHECK:   [[CST_FQ_HI:%.*]] = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.970210e-01> : tensor<1x1x1x1xf16>>
    // CHECK:   [[CONV_WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<64x64x1x1xf16>>

    // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[CONV_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x8x512xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x8x512xf16>

    // CHECK:   [[PRELU_LEFT:%.*]] = IE.LeakyRelu([[CONV]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x64x8x512xf16> -> tensor<1x64x8x512xf16>

    // CHECK:   [[FQ_LEFT:%.*]] = IE.FakeQuantize([[PRELU_LEFT]], [[CST_FQ_LO]], [[CST_FQ_HI]], [[CST_FQ_LO]], [[CST_FQ_HI]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY",
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x64x8x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x8x512xf16>

    // CHECK:   [[PRELU_RIGHT:%.*]] = IE.LeakyRelu(%arg1) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x64x1x512xf16> -> tensor<1x64x1x512xf16>

    // CHECK:   [[FQ_RIGHT:%.*]] = IE.FakeQuantize([[PRELU_RIGHT]], [[CST_FQ_LO]], [[CST_FQ_HI]], [[CST_FQ_LO]], [[CST_FQ_HI]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY",
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x64x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x1x512xf16>

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_LEFT]], [[FQ_RIGHT]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    // CHECK-SAME:  } : tensor<1x64x8x512xf16>, tensor<1x64x1x512xf16> -> tensor<1x64x9x512xf16>

    // CHECK:   return [[CONCAT]] : tensor<1x64x9x512xf16>
}

