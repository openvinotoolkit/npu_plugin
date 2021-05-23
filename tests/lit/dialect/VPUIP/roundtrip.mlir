// RUN: vpux-opt %s | vpux-opt | FileCheck %s

// CHECK-LABEL: @ConstantTensor
func @ConstantTensor(%arg0: memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32> {
    // CHECK: VPUIP.DeclareConstantTensor memref<1x1x28x28xf16> = dense<0.000000e+00> : tensor<1x1x28x28xf32>
    %0 = VPUIP.DeclareConstantTensor memref<1x1x28x28xf16> = dense<0.000000e+00> : tensor<1x1x28x28xf32>
    // CHECK: VPUIP.ConvertUPA {isTrailingSWLayer} inputs(%{{.*}} : memref<1x1x28x28xf16>) outputs(%{{.*}} : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    %1 = VPUIP.ConvertUPA {isTrailingSWLayer} inputs(%0 : memref<1x1x28x28xf16>) outputs(%arg0 : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    return %1 : memref<1x1x28x28xf32>
}
