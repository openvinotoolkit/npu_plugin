// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniquifyReorders
func @UniquifyReorders(%arg0: tensor<1x16x227x227xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %3 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %4 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %5 = IE.And(%3, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>
    
    return %5 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK: [[RESULT:%.*]] = IE.And([[REORDER]], [[REORDER]])
    // CHECK-NOT: IE.Reorder
}


// CHECK-LABEL: @ReordersWithDifferentConsumerOps
func @ReordersWithDifferentConsumerOps(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %cst_69 = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> = #const.Content<dense<0.0> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]>
    %cst_90 = const.Declare tensor<1x16x1x1xf16> = #const.Content<dense<0.0> : tensor<1x16x1x1xf16>>

    %8 = IE.HSwish(%arg0) : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16>
    %9 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %10 = IE.GroupConvolution(%9, %cst_69, %cst_90) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %14 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %15 = IE.Add(%10, %14) {auto_broadcast = "NUMPY"} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    
    return %15 : tensor<1x16x112x112xf16, {order = #NHWC}>
    
    // CHECK-DAG:   [[CST0:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK: IE.HSwish
    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution([[REORDER]], [[CST0]], [[CST1]])
    // CHECK-NOT: IE.Reorder
    // CHECK: IE.Add([[CONV]], [[REORDER]])
}
