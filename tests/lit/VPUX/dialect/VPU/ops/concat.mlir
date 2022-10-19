// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// -----

// CHECK-LABEL: @OneInputFold
func @OneInputFold(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = VPU.Concat(%arg0) { per_axis = {axis = 1} } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: VPU.Concat
    // CHECK:     return %arg0
}
