//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-u8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType0 = !quant.uniform<u8<0:254>:f16:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>
!qElemType2 = !quant.uniform<u8:f16, 1.1534313725490195>
!qElemType3 = !quant.uniform<u8:f16, 2.4627450980392158>

func.func @Conv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType1> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    %2 = IE.Convolution(%1, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14x!qElemType3>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    return %3 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType0> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.ConvertElemType<si8>,
    // CHECK-SAME:      #const.QuantCast<!qElemType1>,
    // CHECK-SAME:      #const.QuantCast<>,
    // CHECK-SAME:      #const.ConvertElemType<i32>,
    // CHECK-SAME:      #const.Add<1.270000e+02 : f64>,
    // CHECK-SAME:      #const.ConvertElemType<ui8>,
    // CHECK-SAME:      #const.QuantCast<!qElemType0>

    // CHECK:       [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    // CHECK:       [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType0> -> tensor<1x3x14x14x!qElemType3>
    // CHECK:       [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL3]]
}
