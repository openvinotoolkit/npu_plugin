// RUN: vpux-opt --split-input-file --convert-avg-pool-to-dw-conv %s | FileCheck %s

// CHECK-LABEL: @ConvertAveragePoolingToGroupConvolution
func @ConvertAveragePoolingToGroupConvolution(%arg0 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x1x1xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [7, 7],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x2048x7x7xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>

    // CHECK-NOT:   IE.AvgPool
    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<2048x1x1x7x7xf16> = #const.Content<dense<2.040100e-02> : tensor<2048x1x1x7x7xf16>>
    // CHECK:       %[[CONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x7x7xf16>, tensor<2048x1x1x7x7xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       return %[[CONV]]
}

