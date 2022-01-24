// RUN: vpux-opt --split-input-file --remove-duplicates %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Reorders
func @Reorders(%arg0: tensor<1x16x227x227xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %3 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %4 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %5 = IE.And(%3, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %5 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK: [[RESULT:%.*]] = IE.And([[REORDER]], [[REORDER]])
    // CHECK-NOT: IE.Reorder
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReordersWithDifferentConsumers
func @ReordersWithDifferentConsumers(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
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

// -----

// CHECK-LABEL: @CopiesInTiling
func @CopiesInTiling(%arg0: tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16> {
    %cst = const.Declare tensor<64x32x3x3xf16> = #const.Content<dense<1.0> : tensor<64x32x3x3xf16>>

    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 32, 65, 128] :
        tensor<1x32x128x128xf16> to tensor<1x32x65x128xf16>
    %1 = IE.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x32x65x128xf16>
        -> tensor<1x32x65x128xf16, {mem_space = @CMX_NN}>
    %2 = IE.Copy(%cst) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16>
        -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>
    %3 = IE.Convolution(%1, %2) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [0, 1],
            strides = [1, 1]
        }
        : tensor<1x32x65x128xf16, {mem_space = @CMX_NN}>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>
        -> tensor<1x64x64x128xf16, {mem_space = @CMX_NN}>

    %4 = IE.Slice %arg0 [0, 0, 63, 0] [1, 32, 65, 128] :
        tensor<1x32x128x128xf16> to tensor<1x32x65x128xf16>
    %5 = IE.Copy(%4) {out_mem_space = @CMX_NN} : tensor<1x32x65x128xf16>
        -> tensor<1x32x65x128xf16, {mem_space = @CMX_NN}>
    %6 = IE.Copy(%cst) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16>
        -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>
    %7 = IE.Convolution(%5, %6) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        }
        : tensor<1x32x65x128xf16, {mem_space = @CMX_NN}>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>
        -> tensor<1x64x64x128xf16, {mem_space = @CMX_NN}>

    %8 = IE.Concat(%3, %7) {per_axis = {axis = 2}}
        : tensor<1x64x64x128xf16, {mem_space = @CMX_NN}>, tensor<1x64x64x128xf16, {mem_space = @CMX_NN}>
        -> tensor<1x64x128x128xf16, {mem_space = @CMX_NN}>
    %9 = IE.Copy(%8) : tensor<1x64x128x128xf16, {mem_space = @CMX_NN}>
        -> tensor<1x64x128x128xf16>
    return %9 : tensor<1x64x128x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<64x32x3x3xf16>

    // CHECK: [[IN_TILE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 32, 65, 128]
    // CHECK: [[IN_TILE0_CMX:%.+]] = IE.Copy([[IN_TILE0]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CST_CMX:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK: [[OUT_TILE0:%.+]] = IE.Convolution([[IN_TILE0_CMX]], [[CST_CMX]])

    // CHECK: [[IN_TILE1:%.+]] = IE.Slice %arg0 [0, 0, 63, 0] [1, 32, 65, 128]
    // CHECK: [[IN_TILE1_CMX:%.+]] = IE.Copy([[IN_TILE1]]) {out_mem_space = @CMX_NN}
    // CHECK: [[OUT_TILE1:%.+]] = IE.Convolution([[IN_TILE1_CMX]], [[CST_CMX]])

    // CHECK: [[OUT_CMX:%.+]] = IE.Concat([[OUT_TILE0]], [[OUT_TILE1]])
    // CHECK: [[OUT:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK: return [[OUT]]
}
