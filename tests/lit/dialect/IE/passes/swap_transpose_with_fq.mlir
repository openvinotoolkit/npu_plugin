// RUN: vpux-opt --split-input-file --swap-transpose-with-fq --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 0.0024337469362745098>

func @SwapTransposeWithPerTensorFQ(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType}
        : tensor<1x70x1x28xf16> -> tensor<1x70x1x28x!qElemType>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28x!qElemType> -> tensor<1x1x28x70x!qElemType>
    return %1 : tensor<1x1x28x70x!qElemType>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   %[[FQ:.*]] = IE.Quantize(%[[TRANSPOSE]])
    // CHECK-SAME:  {dstElemType = !qElemType}
    // CHECK-SAME:  : tensor<1x1x28x70xf16> -> tensor<1x1x28x70x!qElemType>

    // CHECK:   return %[[FQ]] : tensor<1x1x28x70x!qElemType>
}
