// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-reorder-to-permute-quantize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReorder
func @ConvertReorder(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NHWC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Reorder
    // CHECK:       IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = f16,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipNHWCInput
func @SkipNHWCInput(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x3x224x224xf16, {order = #NWHC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NWHC
    } : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x3x224x224xf16, {order = #NWHC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NWHC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipNWHCOutput
func @SkipNWHCOutput(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16, {order = #NWHC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NWHC
    } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NWHC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NWHC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipU8Input
func @SkipU8Input(%arg0: tensor<1x3x224x224xui8>) -> tensor<1x3x224x224xui8, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x224x224xui8> -> tensor<1x3x224x224xui8, {order = #NHWC}>

    return %0 : tensor<1x3x224x224xui8, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipIncompatibleShape
func @SkipIncompatibleShape(%arg0: tensor<1x3x225x225xui8>) -> tensor<1x3x225x225xui8, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x225x225xui8> -> tensor<1x3x225x225xui8, {order = #NHWC}>

    return %0 : tensor<1x3x225x225xui8, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}
