//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --calculate-async-region-cycle-cost  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
!MemRef1 = memref<1x128x64x32xf16, #NWHC>
!Distributed0 = !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed1 = !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed2 = !VPUIP.DistributedBuffer<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!MemRef0 = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
!MemRef2 = memref<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>

// CHECK-LABEL: module @AddCycleCostForSWMultiCluster
module @AddCycleCostForSWMultiCluster attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.ExecutorResource 1 of @DPU
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @SHAVE_NN
        IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }
    IE.ExecutorResource 2 of @DMA_NN 
    IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
    module @VPU.SW {
        func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }
    
    func.func @AddCycleCostForSWMultiClusterTest(%arg0: !MemRef1) -> !MemRef1 {
        %0 = VPURT.AllocDistributed -> !Distributed0
        %1 = VPURT.AllocDistributed -> !Distributed0
        %2 = memref.alloc() : !MemRef1
        %token, %results = async.execute -> !async.value<!Distributed0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
            %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !MemRef1) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !Distributed0 {
            %5 = VPUIP.Copy inputs(%arg1 : !MemRef1) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
            }
            async.yield %4 : !Distributed0
        }
        %token_0, %results_1:2 = async.execute [%token] (%results as %arg1: !async.value<!Distributed0>) -> (!async.value<!Distributed1>, !async.value<!Distributed2>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
            %4 = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
            %5 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
            %6 = VPUIP.SubView %1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
            %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
            %8:2 = VPUIP.NCEClusterTiling inputs(%5 as %arg2: !MemRef0, %4 as %arg3: !MemRef2) outputs(%7 as %arg4: !MemRef0, %6 as %arg5: !MemRef2) -> (!Distributed1, !Distributed2) {
            %results_4:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg2 as %arg6: !MemRef0, %arg3 as %arg7: !MemRef2) outputs(%arg4 as %arg8: !MemRef0, %arg5 as %arg9: !MemRef2) on tile 0 -> (!MemRef0, !MemRef2){
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : !MemRef0, !MemRef0
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg7, %arg9) : !MemRef2, !MemRef2
            }
            }
            async.yield %8#0, %8#1 : !Distributed1, !Distributed2
        }
        %token_2, %results_3 = async.execute [%token_0] (%results_1#0 as %arg1: !async.value<!Distributed1>, %results_1#1 as %arg2: !async.value<!Distributed2>) -> !async.value<!MemRef1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
            %4 = VPUIP.ConcatView inputs(%arg1, %arg2 : !Distributed1, !Distributed2) outputs(%1 : !Distributed0) -> !Distributed0
            %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg3: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%2 as %arg4: !MemRef1) -> !MemRef1 {
            %6 = VPUIP.Copy inputs(%arg3 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg4 : !MemRef1) -> !MemRef1
            }
            async.yield %5 : !MemRef1
        }
        %3 = async.await %results_3 : !async.value<!MemRef1>

        // CHECK:   [[T1:%.+]], [[F1:%.+]] = async.execute -> !async.value<!VPUIP.DistributedBuffer<1x128x64x32xf16
        // CHECK:   async.execute [[[T1]]] ([[F1]] as %arg1: !async.value<!VPUIP.DistributedBuffer<1x128x64x32xf16
        // CHECK-SAME:  VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64, cycleCost = 318646 : i64
        return %3 : !MemRef1
    }
}


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SingleMemRef = memref<4x8x12x16xf16, #NHWC, [@CMX_NN, 0]>
!MemRef0 = memref<4x8x12x16xf16, [@CMX_NN, 0]>
!MemRef1 = memref<4x4x12x16xf16, {order = #NHWC, strides = [1536, 1, 128, 8]}, [@CMX_NN, 0]>
!MemRef2 = memref<4x2x12x16xf16, {order = #NHWC, strides = [1536, 1, 128, 8]}, [@CMX_NN, 0]>

// CHECK-LABEL: module @AddCycleCostForSWSingleCluster
module @AddCycleCostForSWSingleCluster attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 1 of @NCE at 1.300000e+03 MHz {
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @SHAVE_NN
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 1 of @DMA_NN
    
    func.func @AddCycleCostForSWSingleClusterTest(%arg0: memref<4x8x12x16xf16, #NHWC, @DDR>, %arg1: memref<4x8x12x16xf16, @DDR>) -> memref<4x8x12x16xf16, @DDR> {
        %0 = memref.alloc() : !SingleMemRef
        %1 = memref.alloc() : !SingleMemRef
        %2 = memref.alloc() : !MemRef0
        %token, %results = async.execute -> !async.value<!SingleMemRef> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
            %4 = VPUIP.Copy inputs(%arg0 : memref<4x8x12x16xf16, #NHWC, @DDR>) outputs(%0 : !SingleMemRef) -> !SingleMemRef
            async.yield %4 : !SingleMemRef
        }
        %token_0, %results_1:2 = async.execute [%token] (%results as %arg2: !async.value<!SingleMemRef>) -> (!async.value<!MemRef2>, !async.value<!MemRef1>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %4 = VPUIP.SubView %arg2 [0, 0, 0, 0] [4, 2, 12, 16] : !SingleMemRef to !MemRef2
        %5 = VPUIP.SubView %1 [0, 0, 0, 0] [4, 2, 12, 16] : !SingleMemRef to !MemRef2
        %6 = VPUIP.SubView %arg2 [0, 4, 0, 0] [4, 4, 12, 16] : !SingleMemRef to !MemRef1
        %7 = VPUIP.SubView %1 [0, 4, 0, 0] [4, 4, 12, 16] : !SingleMemRef to !MemRef1
        %results_6:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%4 as %arg3: !MemRef2, %6 as %arg4: !MemRef1) outputs(%5 as %arg5: !MemRef2, %7 as %arg6: !MemRef1) on tile 0 -> (!MemRef2, !MemRef1){
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg5) : !MemRef2, !MemRef2
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg4, %arg6) : !MemRef1, !MemRef1
            }
            async.yield %results_6#0, %results_6#1 : !MemRef2, !MemRef1
        }
        %token_2, %results_3 = async.execute [%token_0] (%results_1#0 as %arg2: !async.value<!MemRef2>, %results_1#1 as %arg3: !async.value<!MemRef1>) -> !async.value<!MemRef0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
            %4 = VPUIP.ConcatView inputs(%arg2, %arg3 : !MemRef2, !MemRef1) outputs(%1 : !SingleMemRef) -> !SingleMemRef
            %5 = VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64} inputs(%4 : !SingleMemRef) outputs(%2 : !MemRef0) -> !MemRef0
            async.yield %5 : !MemRef0
        }
        %token_4, %results_5 = async.execute [%token_2] (%results_3 as %arg2: !async.value<!MemRef0>) -> !async.value<memref<4x8x12x16xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
            %4 = VPUIP.Copy inputs(%arg2 : !MemRef0) outputs(%arg1 : memref<4x8x12x16xf16, @DDR>) -> memref<4x8x12x16xf16, @DDR>
            async.yield %4 : memref<4x8x12x16xf16, @DDR>
        }
        %3 = async.await %results_5 : !async.value<memref<4x8x12x16xf16, @DDR>>

        // CHECK:   [[T1:%.+]], [[F1:%.+]] = async.execute -> !async.value<memref<4x8x12x16xf16, #NHWC, [@CMX_NN, 0]>>
        // CHECK:   async.execute [[[T1]]] ([[F1]] as %arg2: !async.value<memref<4x8x12x16xf16, #NHWC, [@CMX_NN, 0]>>
        // CHECK-SAME:  VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64, cycleCost = 10212 : i64
        return %3 : memref<4x8x12x16xf16, @DDR>
    }
}
