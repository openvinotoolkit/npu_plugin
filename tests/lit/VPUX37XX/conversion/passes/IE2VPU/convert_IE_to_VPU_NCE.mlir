//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCE
func @DepthConvToNCE(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = {attrs = {negative_slope = 0.1}, name = "IE.LeakyRelu"}
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, mode = "LPRELU"},
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithReluRewriter
func @EltwiseAddWithReluRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY", post_op = {attrs = {}, name = "IE.ReLU"} } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_scale = [1.000000e+00]}}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolToNCE
func @AveragePoolToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x14x14xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) { 
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = "FLOOR",
            strides = [2, 2]
         } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.AveragePool(%arg0)
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}
    // CHECK-SAME:      -> tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x14x14xf16, {order = #NHWC}>
}
    
// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.049356617647058822>
!qElemType1 = type !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = type !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithDifferentScales(%arg0: tensor<1x64x28x28x!qElemType0, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
func @EltwiseAddWithDifferentScales(%arg0: tensor<1x64x28x28x!qElemType0, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY" } :
        tensor<1x64x28x28x!qElemType0, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:      fp_prelu_alpha = 1.000000e+00 : f64,
    // CHECK-SAME:      in1_quant_mult = [25877], in2_quant_mult = [5312],
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_mult = [19219],
    // CHECK-SAME:      quant_post_shift = 0 : i64, quant_shift = [29]}}
    // CHECK-SAME:      -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseMulToNCE
func @SkipEltwiseMulToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = "NUMPY" } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = "NUMPY"} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseSubToNCE
func @SkipEltwiseSubToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Subtract(%arg0, %arg1) { auto_broadcast = "NUMPY" } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = "NUMPY"} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseAndToNCE
func @SkipEltwiseAndToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg1) { auto_broadcast = "NUMPY" } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.And(%arg0, %arg1) {auto_broadcast = "NUMPY"} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE4channels
func @ConvToNCE4channels(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = {attrs = {negative_slope = 0.1}, name = "IE.LeakyRelu"}
        } : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x4x1x1xf16>,
    // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 4]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, mode = "LPRELU"},
    // CHECK-SAME:      rawFilterShape = [16, 4, 1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.078431372549019607>
!qElemType1 = type !quant.uniform<u8:f16, 0.039215686274509803>

// CHECK-LABEL: @AddWithDifferentScales
func @AddWithDifferentScales(%arg0: tensor<1x16x1x2xui8, {order = #NHWC}>) -> tensor<1x16x1x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x2x!qElemType0, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x16x1x2xf32>, [
            #const.ConvertElemType<f16>,
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType0>,
            #const.Reorder<#NHWC>
        ]

    %0 = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    %1 = IE.Add(%0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x16x1x2x!qElemType1, {order = #NHWC}>, tensor<1x16x1x2x!qElemType0, {order = #NHWC}> -> tensor<1x16x1x2xf16, {order = #NHWC}>

    return %1 : tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x16x1x2x!qElemType0, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x16x1x2xf32>, [
    // CHECK-SAME:    #const.ConvertElemType<f16>,
    // CHECK-SAME:    #const.ConvertElemType<ui8>,
    // CHECK-SAME:    #const.QuantCast<!qElemType0>,
    // CHECK-SAME:    #const.Reorder<#NHWC>
    // CHECK-SAME:  ]

    // CHECK:   [[QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    // CHECK:   [[NCE_ADD:%.*]] = VPU.NCE.Eltwise([[QUANT_CAST]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:     op_type = "ADD",
    // CHECK-SAME:     ppe = {
    // CHECK-SAME:         clamp_high = 2147483647 : i64,
    // CHECK-SAME:         clamp_low = -2147483648 : i64,
    // CHECK-SAME:         fp_prelu_alpha = 1.000000e+00 : f64,
    // CHECK-SAME:         in1_quant_mult = [20560],
    // CHECK-SAME:         in2_quant_mult = [41120],
    // CHECK-SAME:         lrelu_mult = 1 : i64,
    // CHECK-SAME:         lrelu_shift = 0 : i64,
    // CHECK-SAME:         mode = "ADD",
    // CHECK-SAME:         quant_mult = [16384],
    // CHECK-SAME:         quant_post_shift = 0 : i64,
    // CHECK-SAME:         quant_shift = [33]
    // CHECK-SAME:     }
    // CHECK-SAME:  } -> tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK:   return [[NCE_ADD]] : tensor<1x16x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertPermuteQuantize
func @ConvertPermuteQuantize(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK:       [[RESHAPE_IN:%.*]] = VPU.Reshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 224, 3, 224]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x224x3x224xf16>

    // CHECK:       [[LAYOUT_CAST_IN:%.*]] = VPU.LayoutCast([[RESHAPE_IN]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x224x3x224xf16> -> tensor<1x224x3x224xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = VPU.NCE.PermuteQuantize([[LAYOUT_CAST_IN]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NWCH,
    // CHECK-SAME:      pad = {
    // CHECK-SAME:          bottom = 1 : i64,
    // CHECK-SAME:          left = 0 : i64,
    // CHECK-SAME:          right = 0 : i64,
    // CHECK-SAME:          top = 0 : i64
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x224x4x224x!qElemType, {order = #NWCH}>

    // CHECK:       [[LAYOUT_CAST_OUT:%.*]] = VPU.LayoutCast([[PERMUTE_QUANTIZE]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x224x4x224x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x224x4x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[LAYOUT_CAST_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 4, 224, 224]
    // CHECK-SAME:  } : tensor<1x224x4x224x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeDoesNotFitCMX
func @PermuteQuantizeDoesNotFitCMX(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x16x512x512x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x512x512xf16> -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK:       [[RESHAPE_IN:%.*]] = VPU.Reshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 512, 3, 512]
    // CHECK-SAME:  } : tensor<1x3x512x512xf16> -> tensor<1x512x3x512xf16>

    // CHECK:       [[LAYOUT_CAST_IN:%.*]] = VPU.LayoutCast([[RESHAPE_IN]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x512x3x512xf16> -> tensor<1x512x3x512xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = VPU.NCE.PermuteQuantize([[LAYOUT_CAST_IN]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NWCH,
    // CHECK-SAME:      pad = {
    // CHECK-SAME:          bottom = 13 : i64,
    // CHECK-SAME:          left = 0 : i64,
    // CHECK-SAME:          right = 0 : i64,
    // CHECK-SAME:          top = 0 : i64
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x512x16x512x!qElemType, {order = #NWCH}>

    // CHECK:       [[LAYOUT_CAST_OUT:%.*]] = VPU.LayoutCast([[PERMUTE_QUANTIZE]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x512x16x512x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x512x16x512x!qElemType, {order = #NHWC}>

    // CHECK:       [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[LAYOUT_CAST_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 16, 512, 512]
    // CHECK-SAME:  } : tensor<1x512x16x512x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x16x512x512x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeStartPadsOverHeight
func @PermuteQuantizeStartPadsOverHeight(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x225x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 1, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 1, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeEndPadsOverHeight
func @PermuteQuantizeEndPadsOverHeight(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x225x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 1, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 1, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedInputLayout
func @PermuteQuantizeUnsupportedInputLayout(%arg0: tensor<1x3x224x224xf16, {order = #NCWH}>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16, {order = #NCWH}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16, {order = #NCWH}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedOutputLayout
func @PermuteQuantizeUnsupportedOutputLayout(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NWCH}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NWCH,
        mem_perm = #NWCH,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NWCH,
    // CHECK-SAME:      mem_perm = #NWCH,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NWCH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedShape
func @PermuteQuantizeUnsupportedShape(%arg0: tensor<1x3x225x225xf16>) -> tensor<1x4x225x225x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x225x225xf16> -> tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x225x225xf16> -> tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x225x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ClampLowInF16PReLU(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%arg0, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        post_op = {
            attrs = {
                negative_slope = -2.312500e+00 : f64
            },
            name = "IE.LeakyRelu"
        },
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:   clamp_low = -2147483648 : i64

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>
}
