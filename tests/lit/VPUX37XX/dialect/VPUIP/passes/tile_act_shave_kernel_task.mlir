//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --tile-act-shave-kernel-task --canonicalize %s | FileCheck %s

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @TileStridedMVN(%arg0: memref<1x128x64x32xf16, #NWHC>)
        -> memref<1x128x64x32xf16, #NWHC> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs(%0 : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    return %4: memref<1x128x64x32xf16, #NWHC>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>

    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16, #NWHC>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func @TileClusterStridedMVN(%arg0: memref<1x128x64x32xf16, #NWHC>)
        -> memref<1x128x64x32xf16, #NWHC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NWHC>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x64x32xf16, #NWHC, @CMX_NN>, memref<1x128x64x32xf16, #NWHC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    }
    return %6: memref<1x128x64x32xf16, #NWHC>


    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
    // CHECK:      %results:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
    // CHECK:      }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:    }
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x128x64x32xf16, #NWHC>
}
