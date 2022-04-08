//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --fuse-clamp --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp32to128
func @QuantClamp32to128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 128 : i64,
    // CHECK-SAME:          clamp_low = 32 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          mode = "LPRELU"
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SkipClampWithoutNCE
func @SkipClampWithoutNCE(%arg0: tensor<1x255x56x56x!qElemType, {order = #NHWC}>)
                          -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x255x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.+]] = VPU.Expand
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[EXPAND]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SkipNCEWithNoAttr
func @SkipNCEWithNoAttr(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[ELTWISE]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16:1, {0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0}>

// CHECK-LABEL: @SkipPerChannelQuant
func @SkipPerChannelQuant(%arg0: tensor<1x16x56x56x!qElemType, {order = #NHWC}>,
                          %arg1: tensor<16x16x1x1x!qElemType, {order = #NHWC}>)
                          -> tensor<1x16x56x56x!qElemType, {order = #NHWC}> {
    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %WEIGHT_TABLE) {
        pad = {
            bottom = 0 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        },
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1638 : i64,
            lrelu_shift = 14 : i64,
            mode = "LPRELU"
        },
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x16x56x56x!qElemType, {order = #NHWC}> -> tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[CONV]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipFloat16WithNonZeroMin
func @SkipFloat16WithNonZeroMin(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                                %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                                -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[ELTWISE]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to120
func @FloatClamp0to120(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.200000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 22400 : i64,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          mode = "LRELUX"
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to128
func @FloatClamp0to128(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 22400 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 22400 : i64,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          mode = "LRELUX"
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8<0:255>:f16, 0.5:127>

// CHECK-LABEL: @QuantClampNeg6to6
func @QuantClampNeg6to6(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 6.000000e+00 : f64,
        min = -6.000000e+00 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 139 : i64,
    // CHECK-SAME:          clamp_low = 115 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          mode = "LPRELU"
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp0To128
func @QuantClamp0To128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                       -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 120 : i64,
            clamp_low = 32 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = "LPRELU"
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 120 : i64,
    // CHECK-SAME:          clamp_low = 32 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          mode = "LPRELU"
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}
