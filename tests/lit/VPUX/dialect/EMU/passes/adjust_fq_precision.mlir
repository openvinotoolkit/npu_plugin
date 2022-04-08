//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-fq-precision %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @AdjustFQPrecision
func @AdjustFQPrecision(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %output_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %fq = VPU.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %fq : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[ILOW:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   %[[IHIGH:.*]] = const.Declare tensor<f16> = dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   %[[OLOW:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   %[[OHIGH:.*]] = const.Declare tensor<f16> = dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]


    // CHECK:       %[[FQ:.*]] = VPU.FakeQuantize(%arg0, %[[ILOW]], %[[IHIGH]], %[[OLOW]], %[[OHIGH]])
    // CHECK:       return %[[FQ]]
}

// CHECK-LABEL: @AdjustFQPrecisionAliased
func @AdjustFQPrecisionAliased(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %fq = VPU.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %fq : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[LOW:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   %[[HIGH:.*]] = const.Declare tensor<f16> = dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]

    // CHECK:       %[[FQ:.*]] = VPU.FakeQuantize(%arg0, %[[LOW]], %[[HIGH]], %[[LOW]], %[[HIGH]])
    // CHECK:       return %[[FQ]]
}
