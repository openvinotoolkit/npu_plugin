//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --unroll-sw-kernel %s | FileCheck %s

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @UnrollSwKernel()
        -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
        %results:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%2 as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %4 as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 as %arg2: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, %5 as %arg3: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>){
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg2) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
        }
    }
    return %3, %5: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
    // CHECK:           VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs([[TILE0]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT0]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
    // CHECK:           VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs([[TILE1]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT1]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg0, %arg1) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @UnrollClusterSwKernel()
        -> (memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) {

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
         %6:2 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: memref<1x64x64x32xf16, #NWHC, @CMX_NN>, %4 as %arg1: memref<1x64x64x32xf16, #NWHC, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x64x32xf16, #NWHC, @CMX_NN>, %5 as %arg3: memref<1x64x64x32xf16, #NWHC, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
              %results:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg0 as %arg4: memref<1x64x64x32xf16, #NWHC, @CMX_NN>, %arg1 as %arg5: memref<1x64x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 as %arg6: memref<1x64x64x32xf16, #NWHC, @CMX_NN>, %arg3 as %arg7: memref<1x64x64x32xf16, #NWHC, @CMX_NN>) on tile 0 -> (memref<1x64x64x32xf16, #NWHC, @CMX_NN>, memref<1x64x64x32xf16, #NWHC, @CMX_NN>){
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg4, %arg6) : memref<1x64x64x32xf16, #NWHC, @CMX_NN>, memref<1x64x64x32xf16, #NWHC, @CMX_NN>
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, #NWHC, @CMX_NN>, memref<1x64x64x32xf16, #NWHC, @CMX_NN>
              }
        }
    }
    return %3, %5: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>


    // CHECK:   [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[TILE0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <663616> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <925760> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[TILE1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1187904> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1450048> -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
    // CHECK:       VPUIP.NCEClusterTiling inputs([[TILE0]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT0]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]> {
    // CHECK:              VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg0 as %arg2: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%arg1 as %arg3: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                  VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:              }
    // CHECK:       }
    // CHECK:   }
    // CHECK:   VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 1000 : i64, isTrailingSWLayer = false} {
    // CHECK:       VPUIP.NCEClusterTiling inputs([[TILE1]] as %arg0: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT1]] as %arg1: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]> {
    // CHECK:              VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg0 as %arg2: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%arg1 as %arg3: memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>{
    // CHECK:                  VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:              }
    // CHECK:       }
    // CHECK:   }
    // CHECK:   return [[OUTPUT0]], [[OUTPUT1]] : memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x64x64x32xf16, #NWHC, [@CMX_NN, 0]>
}
