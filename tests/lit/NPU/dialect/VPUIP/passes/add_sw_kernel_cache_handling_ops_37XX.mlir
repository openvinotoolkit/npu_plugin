//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-sw-kernel-cache-handling-ops %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
 func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_relu.cpp",
            VPU.kernel_entry = "activation_relu",
            VPU.task_type = @COMPUTE
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @AddCacheHandlingSwOp
func.func @AddCacheHandlingSwOp(%1: memref<1x1x1x1000xf16, @DDR>, %2: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr as %arg0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg0, %arg1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) outputs(%2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    return %2: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
    // CHECK:   [[DECLARE_BUFFER_1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DECLARE_BUFFER_2:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   [[BARRIER_3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_relu inputs([[DECLARE_BUFFER_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:           VPUIP.SW.Kernel.run
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   return %arg1 : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
 func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_relu.cpp",
            VPU.kernel_entry = "activation_relu",
            VPU.task_type = @COMPUTE
        }

func.func private @builtin_hswish(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "hswish_fp16.cpp",
            VPU.kernel_entry = "hswish_fp16",
            VPU.task_type = @COMPUTE
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @AAddCacheHandling2SwOps
func.func @AddCacheHandling2SwOps(%1: memref<1x1x1x1000xf16, @DDR>, %2: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr as %arg0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg0, %arg1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b1  : !VPURT.Barrier) updates(%b2  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_ddr as %arg0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg0, %arg1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) outputs(%2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    return %2: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
    // CHECK:   [[DECLARE_BUFFER_1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DECLARE_BUFFER_2:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   [[BARRIER_4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_4]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_relu inputs([[DECLARE_BUFFER_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_4]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     }
    // CHECK:   }
    // CHECK:   [[BARRIER_5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_5]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_hswish inputs([[DECLARE_BUFFER_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_5]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:     VPUIP.NNDMA inputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   return %arg1 : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
 func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_relu.cpp",
            VPU.kernel_entry = "activation_relu"
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @AddCacheHandlingSwOp
func.func @AddCacheHandlingSwOpNoTaskType(%1: memref<1x1x1x1000xf16, @DDR>, %2: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr as %arg0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg0, %arg1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) outputs(%2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    return %2: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
    // CHECK:   [[DECLARE_BUFFER_1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DECLARE_BUFFER_2:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   [[BARRIER_3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_relu inputs([[DECLARE_BUFFER_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[DECLARE_BUFFER_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:           VPUIP.SW.Kernel.run
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs([[DECLARE_BUFFER_2]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   }
    // CHECK:   return %arg1 : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
 func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_relu.cpp",
            VPU.kernel_entry = "activation_relu",
            VPU.task_type = @COMPUTE
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @AddUpdateBarrierToMultiKernelSwKernel
func.func @AddUpdateBarrierToMultiKernelSwKernel(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>, %arg2: memref<1x1x1x1000xf16, @DDR>, %arg3: memref<1x1x1x1000xf16, @DDR>)
        -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %in_ddr_1  = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_1 = VPURT.DeclareBuffer <DDR> <6000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr_0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %in_ddr_1 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr_0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %out_ddr_1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr_0 : memref<1x1x1x1000xf16, @DDR>) outputs(%arg2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr_1 : memref<1x1x1x1000xf16, @DDR>) outputs(%arg3 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    return %arg2, %arg3: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:       [[IN_BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[IN_BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[OUT_BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[OUT_BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <6000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:      VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_relu
    // CHECK:       inputs([[IN_BUFF_0]] as %arg4: memref<1x1x1x1000xf16, @DDR>, [[IN_BUFF_1]] as %arg5: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       outputs([[OUT_BUFF_0]] as %arg6: memref<1x1x1x1000xf16, @DDR>, [[OUT_BUFF_1]] as %arg7: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>){
    // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg4, %arg6) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK-NEXT:      VPUIP.SW.Kernel.run
    // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.NNDMA inputs([[OUT_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.NNDMA inputs([[OUT_BUFF_1]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg3 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       return %arg2, %arg3 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @cache_flush()
        attributes {
            VPU.task_type = @CACHE_FLUSH
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @DontAddCacheHandlingSwOp
func.func @DontAddCacheHandlingSwOp(%1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

   VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0{
            VPUIP.SW.Kernel.run
        }
    }

    return %1 : memref<1x1x1x1000xf16, @DDR>

    // CHECK-NOT:   func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>)
    attributes {
        VPU.kernel_code = "single_shave_convert.cpp",
        VPU.kernel_entry = "single_shave_convert"
    }

func.func private @builtin_Gather(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "single_shave_gather.cpp",
        VPU.kernel_entry = "single_shave_gather",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

// CHECK-LABEL @AddCacheInvalidateSwOpForDDRInputCMXOutput
func.func @AddCacheInvalidateSwOpForDDRInputCMXOutput(%arg0: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %cst = const.Declare memref<51865x512xf16> = dense<1.0> : tensor<51865x512xf32>, [#const.ConvertElemType<f16>]

    %indices_fp16_ddr = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1xf16, @DDR>
    %indices_fp16_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457216> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    %indices_si32_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    %indices_input_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1xsi32, [@CMX_NN, 0]>

    %output_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1473024> -> memref<1x1x512xf16, [@CMX_NN, 0]>
    %output_ddr = VPURT.DeclareBuffer <DDR> <100> -> memref<1x1x512xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%indices_fp16_ddr : memref<1x1x1x1xf16, @DDR>) outputs(%indices_fp16_cmx : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs(%indices_fp16_cmx as %arg2: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs(%indices_si32_cmx as %arg3: memref<1x1x1x1xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x1xsi32, [@CMX_NN, 0]>
        }
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Gather inputs(%cst as %arg2: memref<51865x512xf16>, %indices_input_cmx as %arg3: memref<1x1xsi32, [@CMX_NN, 0]>) outputs(%output_cmx as %arg4: memref<1x1x512xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x512xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg4) : memref<51865x512xf16>, memref<1x1xsi32, [@CMX_NN, 0]>, memref<1x1x512xf16, [@CMX_NN, 0]>
        }
    }
    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%output_cmx : memref<1x1x512xf16, [@CMX_NN, 0]>) outputs(%output_ddr : memref<1x1x512xf16, @DDR>) -> memref<1x1x512xf16, @DDR>
    }

    return %arg0: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_INVALIDATE}

    // CHECK:   [[CONST:%.*]] = const.Declare memref<51865x512xf16> = dense<1.000000e+00> : tensor<51865x512xf32>, [#const.ConvertElemType<f16>]
    // CHECK:   [[INDICES_FP16_DDR_BUF:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1xf16, @DDR>
    // CHECK:   [[INDICES_FP16_CMX_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457216> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_SI32_CMX_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_CMX_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1xsi32, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_CMX_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1473024> -> memref<1x1x512xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR_BUF:%.*]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x1x512xf16, @DDR>

    // CHECK:   [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BARRIER_3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs([[INDICES_FP16_DDR_BUF]] : memref<1x1x1x1xf16, @DDR>) outputs([[INDICES_FP16_CMX_BUF]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   }

    // CHECK:   VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[INDICES_FP16_CMX_BUF]] as %arg1: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[INDICES_SI32_CMX_BUF]] as %arg2: memref<1x1x1x1xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>{
    // CHECK:           VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:       }
    // CHECK:   }

    // CHECK:   [[BARRIER_4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_4]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Gather inputs([[CONST]] as %arg1: memref<51865x512xf16>, [[INDICES_CMX_BUF]] as %arg2: memref<1x1xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX_BUF]] as %arg3: memref<1x1x512xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x512xf16, [@CMX_NN, 0]>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg1, %arg2, %arg3) : memref<51865x512xf16>, memref<1x1xsi32, [@CMX_NN, 0]>, memref<1x1x512xf16, [@CMX_NN, 0]>
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits([[BARRIER_4]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:           VPUIP.SW.Kernel.run
    // CHECK:       }
    // CHECK:   }

    // CHECK:   VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NNDMA inputs([[OUTPUT_CMX_BUF]] : memref<1x1x512xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR_BUF]] : memref<1x1x512xf16, @DDR>) -> memref<1x1x512xf16, @DDR>
    // CHECK:   }

    // CHECK:   return %arg0 : memref<1x1x1x1000xf16, @DDR>
}
