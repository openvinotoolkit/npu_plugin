// RUN: vpux-opt --split-input-file --dequantize-const %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f32:0, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @PerAxis
func @PerAxis() -> tensor<4x1x1x1xf32> {
    %0 = const.Declare
        tensor<4x1x1x1x!quant.uniform<u8:f32:0, {0.1:128, 0.2:128, 0.3:128, 0.4:128}>> =
            #const.Content<dense<129> : tensor<4x1x1x1xui8>,
                [#const.QuantCast<!quant.uniform<u8:f32:0, {0.1:128, 0.2:128, 0.3:128, 0.4:128}>>]>

    %1 = "quant.dcast"(%0) :
        (tensor<4x1x1x1x!quant.uniform<u8:f32:0, {0.1:128, 0.2:128, 0.3:128, 0.4:128}>>)
            -> tensor<4x1x1x1xf32>

    return %1 : tensor<4x1x1x1xf32>

    // CHECK:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      #const.Content<dense<129> : tensor<4x1x1x1xui8>
    // CHECK-SAME:      #const.QuantCast<!qElemType>
    // CHECK-SAME:      #const.Dequantize

    // CHECK:       return [[CST]]
}
