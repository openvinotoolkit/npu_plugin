// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --adjust-layouts --canonicalize %s | FileCheck %s

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @CMajorConv
module @CMajorConv attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
func @main(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<16x3x1x1xf16> = #const.Content<dense<1.0> : tensor<16x3x1x1xf16>>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x32x32xf16>, tensor<16x3x1x1xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC}
    // CHECK-SAME:       -> tensor<1x3x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Convolution([[VAR0]], [[CST]])
    // CHECK-SAME:       -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW}
    // CHECK:       return [[VAR2]] : tensor<1x16x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMajorConvCase2
module @CMajorConvCase2 attributes {VPU.compilationMode = "DefaultHW"} {

func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x3x3x3xf16> = #const.Content<dense<1.0> : tensor<48x3x3x3xf16>>

    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62xf16>

    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x62x62xf16>, tensor<48x3x3x3xf16> -> tensor<1x48x60x60xf16>

    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>

    return %2 : tensor<1x48x60x60xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.+]] = IE.Convolution(%arg0, [[CST]])
    // CHECK-SAME:       -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x48x60x60xf16, {order = #NHWC}>
}

}
