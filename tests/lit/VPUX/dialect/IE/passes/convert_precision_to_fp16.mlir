//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16 --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

//
// The 'convert-precision-to-fp16' pass:
//
//   * Updates both Function bodies and Function prototypes.
//   * It shouldn't touch user types defined in `IE.CNNNetwork`.
//   * It should update types for `Constant` operation.
//

// CHECK-LABEL: @FP32toFP16
module @FP32toFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[OUT:.*]] = IE.SoftMax(%arg0)
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: DataInfo "output" : tensor<1x2x2x2xf32>
        DataInfo "output" : tensor<1x2x2x2xf32>
    }

// CHECK: func.func @main() -> tensor<1x2x2x2xf16>
func.func @main() -> tensor<1x2x2x2xf32> {
    %0 = const.Declare tensor<1x2x2x2xf32> = dense<1.0> : tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>

    // CHECK-DAG:       %[[OUT:.*]] = const.Declare tensor<1x2x2x2xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]
    // CHECK:       return %[[OUT]] : tensor<1x2x2x2xf16>
}

}

// -----

// CHECK-LABEL: @I8ToFp16
module @I8ToFp16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "in1" : tensor<1xf16>
        // CHECK: DataInfo "in2" : tensor<1xf16>
        DataInfo "in1" : tensor<1xf16>
        DataInfo "in2" : tensor<1xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1xf16>
        DataInfo "out" : tensor<1xf16>
    }

// CHECK: func.func @main(%arg0: tensor<1xf16>, %arg1: tensor<1xf16>) -> tensor<1xf16>
func.func @main(%arg0: tensor<1xf16>, %arg1: tensor<1xf16>) -> tensor<1xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = i8} : tensor<1xf16> -> tensor<1xi8>
    %1 = IE.Convert(%arg1) {dstElemType = i8} : tensor<1xf16> -> tensor<1xi8>
    %2 = IE.And(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xi8>, tensor<1xi8> -> tensor<1xi8>
    %3 = IE.Convert(%2) {dstElemType = f16} : tensor<1xi8> -> tensor<1xf16>
    return %3 : tensor<1xf16>

    // CHECK:  %0 = IE.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
    // CHECK:  return %0 : tensor<1xf16>
}

}

// -----

// CHECK-LABEL: @OneHot
module @OneHot {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_2994" : tensor<4xsi32>
        DataInfo "Parameter_2994" : tensor<4xsi32>
    }
    outputsInfo : {
        // CHECK: DataInfo "OneHot_2998" : tensor<3x4xf32>
        DataInfo "OneHot_2998" : tensor<3x4xf32>
    }

// CHECK: func.func @main(%arg0: tensor<4xsi32>) -> tensor<3x4xf16>
func.func @main(%arg0: tensor<4xsi32>) -> tensor<3x4xf32> {
    %0 = IE.OneHot(%arg0) {axis_attr = 0 : i64, depth_attr = 3 : i64, off_value_attr = 0.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, outElemType = f32} : tensor<4xsi32> -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>

    // CHECK:       %[[OUT:.*]] = IE.OneHot(%arg0)
    // CHECK-SAME:      tensor<4xsi32> -> tensor<3x4xf16>
    // CHECK: return %[[OUT]] : tensor<3x4xf16>
}

}
