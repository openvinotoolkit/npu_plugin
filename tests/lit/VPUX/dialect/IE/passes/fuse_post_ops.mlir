// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-post-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func @Conv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @MaxPoolWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = "CEIL"
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.MaxPool
    // CHECK-SAME:     kernel_size = [2, 2]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     rounding_type = "CEIL"
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @DepthWiseConv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x1x2x2xf16> = #const.Content<dense<1.0> : tensor<16x1x1x2x2xf16>, [#const.Reshape<[16, 1, 2, 2]>]>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            groups = 16,
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0]
        } :
        tensor<1x16x4x4xf16>, tensor<16x1x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     groups = 16
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @Conv2dWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.Clamp
}

// -----

func @AddWithReLUTest() -> tensor<1x16x4x4xf16> {
    %0 = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<6.0> : tensor<1x16x4x4xf16>>
    %1 = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<-7.0> : tensor<1x16x4x4xf16>>
    %sum = IE.Add(%0, %1) { auto_broadcast = "NUMPY" } : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %relu = IE.ReLU(%sum) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    return %relu : tensor<1x16x4x4xf16>

    // CHECK:       %[[RIGHT:.*]] = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<-7.000000e+00> : tensor<1x16x4x4xf16>>
    // CHECK:       %[[LEFT:.*]] = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<6.000000e+00> : tensor<1x16x4x4xf16>>
    // CHECK:       %[[SUM:.*]] = IE.Add(%[[LEFT]], %[[RIGHT]])
    // CHECK-SAME:     auto_broadcast = "NUMPY"
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-NOT:   IE.ReLU
}

// -----

func @AddWithLeakyReluTest() -> tensor<1x16x4x4xf16> {
    %0 = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<6.0> : tensor<1x16x4x4xf16>>
    %1 = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<-7.0> : tensor<1x16x4x4xf16>>
    %sum = IE.Add(%0, %1) { auto_broadcast = "NUMPY" } : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %leakyRelu = IE.LeakyRelu(%sum) {
            negative_slope = 0.100000e+00
        } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    return %leakyRelu : tensor<1x16x4x4xf16>

    // CHECK:       %[[RIGHT:.*]] = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<-7.000000e+00> : tensor<1x16x4x4xf16>>
    // CHECK:       %[[LEFT:.*]] = const.Declare tensor<1x16x4x4xf16> = #const.Content<dense<6.000000e+00> : tensor<1x16x4x4xf16>>
    // CHECK:       %[[SUM:.*]] = IE.Add(%[[LEFT]], %[[RIGHT]])
    // CHECK:   IE.LeakyRelu
}

// -----

func @ShouldNotFuseScaleShiftTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %bias = const.Declare tensor<1x16x1x1xf32> = #const.Content<dense<3.0> : tensor<1x16x1x1xf32>>
    %1 = IE.ScaleShift(%0, %bias)
        {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x16x3x3xf16>, tensor<1x16x1x1xf32> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:   IE.Convolution
    // CHECK:   IE.ScaleShift
}

// -----

func @Conv2dWithSigmoidNotFusedTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Sigmoid(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-NOT:     post_op = {attrs = {}, name = "IE.Sigmoid"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NEXT:   IE.Sigmoid
}

// -----

func @Conv2dWithSigmoidTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %cst = const.Declare tensor<f16> = #const.Content<dense<1.270000e+02> : tensor<f16>>
    %cst_0 = const.Declare tensor<f16> = #const.Content<dense<-1.280000e+02> : tensor<f16>>
    %cst_1 = const.Declare tensor<f16> = #const.Content<dense<6.000000e+00> : tensor<f16>>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
         {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Sigmoid(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>
    
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.Sigmoid"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.Sigmoid
}

// -----

func @Conv2dWithTanhNotFusedTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Tanh(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-NOT:     post_op = {attrs = {}, name = "IE.Tanh"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NEXT:   IE.Tanh
}

// -----

func @Conv2dWithTanhTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %cst = const.Declare tensor<f16> = #const.Content<dense<1.270000e+02> : tensor<f16>>
    %cst_0 = const.Declare tensor<f16> = #const.Content<dense<-1.280000e+02> : tensor<f16>>
    %cst_1 = const.Declare tensor<f16> = #const.Content<dense<6.000000e+00> : tensor<f16>>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
         {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Tanh(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>
    
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.Tanh"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.Tanh
}

// -----

func @Conv2dWithLeakyReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.000000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @Conv2dWithLeakyRelu15Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.500000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @MaxPoolWithLeakyReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = "CEIL"
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.MaxPool
    // CHECK-SAME:     kernel_size = [2, 2]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.000000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     rounding_type = "CEIL"
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}
