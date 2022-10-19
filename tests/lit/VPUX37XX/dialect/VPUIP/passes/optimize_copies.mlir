// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --optimize-copies %s | FileCheck %s

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @OptimizeLastCopy(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>, %arg2: memref<1x2x4x4xf16>)
                            -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x4x4xf16>>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0: memref<1x2x4x4xf16>) outputs(%1 : memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
        ^bb0(%arg3: memref<1x2x4x4xf16>, %arg4: memref<1x2x4x4xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %4 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%0: memref<1x2x4x4xf16>) outputs(%2 : memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
        ^bb0(%arg3: memref<1x2x4x4xf16>, %arg4: memref<1x2x4x4xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }

    %5 = VPUIP.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK: [[VAR0:%.*]] = const.Declare

    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>

    // CHECK: [[VAR1:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>)
    // CHECK-SAME:           -> memref<1x2x4x4xf16>
    // CHECK: [[VAR2:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR0]] : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>)
    // CHECK-SAME:           -> memref<1x2x4x4xf16>
    // CHECK: return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2 * 50 + d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @NoChangesTypeMismatch(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50x1x1xf16, #NHWC, #map>) -> memref<1x50x1x1xf16, #NHWC, #map> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0: memref<1x50x1x1xf16>) outputs(%0 : memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
        ^bb0(%arg2: memref<1x50x1x1xf16>, %arg3: memref<1x50x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16, #NHWC, #map>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50x1x1xf16, #NHWC, #map>) outputs(%arg1 : memref<1x50x1x1xf16, #NHWC, #map>) -> memref<1x50x1x1xf16, #NHWC, #map>
    return %3 : memref<1x50x1x1xf16, #NHWC, #map>

    // CHECK:       memref.alloc()
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       VPUIP.PermuteCast
    // CHECK:       [[VAR0:%.*]] = VPUIP.Copy
    // CHECK:       return [[VAR0]]
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @NoChangesInputIsBlockArgument(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                                    %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>) ->
                                    (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    %1 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0: memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
        ^bb0(%arg4: memref<1x2x4x4xf16>, %arg5: memref<1x2x4x4xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %2 = VPUIP.Copy inputs(%1 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %0, %1, %2 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.Copy
    // CHECK:       [[VAR1:%.*]] = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy
    // CHECK:       return [[VAR0]], [[VAR1]], [[VAR2]]
}
