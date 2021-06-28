// RUN: vpux-opt --split-input-file --merge-fake-quant %s | FileCheck %s

// CHECK-LABEL: @PerTensor
func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = "quant.qcast"(%arg0) : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<u8:f32, 1.0:0>>
    %1 = "quant.dcast"(%0) : (tensor<1x4x!quant.uniform<u8:f32, 1.0:0>>) -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>

    // CHECK:       [[MIN:%.*]] = const.Declare tensor<f32> = #const.Content<dense<0.000000e+00> : tensor<f32>>
    // CHECK:       [[MAX:%.*]] = const.Declare tensor<f32> = #const.Content<dense<2.550000e+02> : tensor<f32>>

    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[MIN]], [[MAX]], [[MIN]], [[MAX]])
    // CHECK-SAME:      levels = 256

    // CHECK:       return [[FQ]]
}
