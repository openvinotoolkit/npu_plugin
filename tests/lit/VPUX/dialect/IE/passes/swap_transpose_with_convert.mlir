// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-transpose-with-convert --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[TRANSPOSE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 1.0>

func @DoNotSwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>
    %1 = IE.Add(%0, %0) { auto_broadcast = "NUMPY" } :
        tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>

    %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %3 : tensor<1x1x28x70xf16>

    // CHECK:   %[[VAR0:.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   %[[ADD:.*]] = IE.Add(%[[VAR0]], %[[VAR0]]) {auto_broadcast = "NUMPY"}
    // CHECK-SAME:  : tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[ADD]]) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>
    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%[[CONVERT]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[TRANSPOSE]] : tensor<1x1x28x70xf16>
}
