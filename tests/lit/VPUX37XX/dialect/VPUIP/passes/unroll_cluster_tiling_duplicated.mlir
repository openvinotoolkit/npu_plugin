//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-cluster-tiling  %s | FileCheck %s


// SW Layer Unrolling
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!typeCmxDistributed = !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCWH, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!type_CMX_memref = memref<1x4x512x1xf16, #NCWH, @CMX_NN>
!type_CMX_tensor = tensor<1x4x512x512xf16, {mem_space = @CMX_NN, order = #NCWH}>


!Input_DDR  = memref<1x4x512x1xf16, #NCWH, @DDR>
!Output_DDR = memref<1x4x512x1xf16, #NCWH, @DDR>


VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


// NCEClusterTiling operation with distributed type of input, and output - splitting strategy for SOK intended for full activation
// CHECK-LABEL: @UnrollSWOpInterface
func.func @UnrollSWOpInterface(%input0: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %395 = VPURT.DeclareBuffer <DDR> <4096> -> !Input_DDR
    %302 = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR

    %300 = VPURT.DeclareBuffer <CMX_NN> <0> -> !typeCmxDistributed
    %301 = VPURT.DeclareBuffer <CMX_NN> <4096> -> !typeCmxDistributed


    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1907728 : i64, cycleEnd = 1907899 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%395 as %arg2: !Input_DDR) outputs(%300 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !Input_DDR) outputs(%arg3 : !type_CMX_memref) -> !type_CMX_memref
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%300 as %arg2: !type_CMX_memref) outputs(%301 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
        %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
          @VPU.SW::@builtin_MVN inputs(%arg2 as %arg4: !type_CMX_memref) outputs(%arg3 as %arg5: !type_CMX_memref) on tile 0 -> !type_CMX_memref {
          VPUIP.SW.Kernel.run {
            attrs = [false, true, 1.0013580322265625E-5]}(%arg4, %arg5) : !type_CMX_memref, !type_CMX_memref
        }
      }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  attributes {cycleBegin = 1907901 : i64, cycleEnd = 1908072 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%301 as %arg2 : !type_CMX_memref) outputs(%302 as %arg3: !Output_DDR) -> !Output_DDR {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !type_CMX_memref) outputs(%arg3 : !Output_DDR) -> !Output_DDR
      }
    }


    return %output: !Output_DDR
}


//CHECK:    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
//CHECK:       %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%4 as %arg2: memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%8 as %arg3: memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 0]>{
//CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 0]>
//CHECK:      }
//CHECK:    }

//CHECK:    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
//CHECK:       %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%5 as %arg2: memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 1]>) outputs(%9 as %arg3: memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 1]>) on tile 1 -> memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 1]>{
//CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 1]>, memref<1x4x512x1xf16, #NCWH, [@CMX_NN, 1]>
//CHECK:      }
//CHECK:    }
