// RUN: vpux-opt --split-input-file %s

func @ConstantTensor(%arg0: memref<1x1x28x28xf32>) {
    %0 = VPUIP.DeclareConstantTensor memref<1x1x28x28xf16> = dense<0.000000e+00> : tensor<1x1x28x28xf32>
    VPUIP.ConvertUPA {isTrailingSWLayer} inputs(%0 : memref<1x1x28x28xf16>) outputs(%arg0 : memref<1x1x28x28xf32>)
    return
}
