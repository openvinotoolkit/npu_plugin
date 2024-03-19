//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file  --init-compiler="vpu-arch=%arch%" --act-shave-profiling %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: @ActShaveProfiling
module @ActShaveProfiling {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x3x224x224xf32>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x150528xf32>
    } profilingOutputsInfo :  {
    }

    VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "softmaxx.cpp", VPU.kernel_entry = "softmax"}
        func.func private @builtin_ConvertF32F16(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"}
        func.func private @builtin_ConvertF16F32(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x150528xf32>) -> memref<1x150528xf32> {
        %0 = memref.alloc() : memref<1x3x224x224xf16, @DDR>
        %1 = memref.alloc() : memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x224x224xf32>) outputs(%1 : memref<1x3x224x224xf32, [@CMX_NN, 0]>) -> memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %3 = memref.alloc() : memref<1x3x224x224xf16, [@CMX_NN, 0]>
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_ConvertF32F16 inputs(%2 as %arg2: memref<1x3x224x224xf32, [@CMX_NN, 0]>) outputs(%3 as %arg3: memref<1x3x224x224xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x224x224xf16, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x3x224x224xf32, [@CMX_NN, 0]>, memref<1x3x224x224xf16, [@CMX_NN, 0]>
        }
        %4 = VPUIP.Copy inputs(%results : memref<1x3x224x224xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        %5 = VPUIP.GenericReshape inputs(%4 : memref<1x3x224x224xf16, @DDR>) -> memref<1x150528xf16, @DDR>
        %6 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %7 = VPUIP.Copy inputs(%5 : memref<1x150528xf16, @DDR>) outputs(%6 : memref<1x150528xf16, [@CMX_NN, 0]>) -> memref<1x150528xf16, [@CMX_NN, 0]>
        %8 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %results_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs(%7 as %arg2: memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%8 as %arg3: memref<1x150528xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf16, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf16, [@CMX_NN, 0]>
        }
        %9 = memref.alloc() : memref<1x150528xf32, [@CMX_NN, 0]>
        %results_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_ConvertF16F32 inputs(%results_0 as %arg2: memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%9 as %arg3: memref<1x150528xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf32, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf32, [@CMX_NN, 0]>
        }
        %10 = VPUIP.Copy inputs(%results_1 : memref<1x150528xf32, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x150528xf32>) -> memref<1x150528xf32>
        return %10 : memref<1x150528xf32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<12xui32>
    //CHECK:        func.func @main(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x150528xf32>, %arg2: memref<12xui32>) -> (memref<1x150528xf32>, memref<12xui32>)
    //CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<12xui32, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.+]] = VPUIP.SubView [[VAR0]] [0] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_ConvertF32F16
    //CHECK-SAME:   profiling_data([[VAR1]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR2:%.+]] = VPUIP.SubView [[VAR0]] [4] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_SoftMax
    //CHECK-SAME:   profiling_data([[VAR2]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR3:%.+]] = VPUIP.SubView [[VAR0]] [8] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_ConvertF16F32
    //CHECK-SAME:   profiling_data([[VAR3]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        return [[R1:%.+]], [[R2:%.+]] : memref<1x150528xf32>, memref<12xui32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!type_CMX = memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
!type_CMX_subview = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>

!type_DDR = memref<1x128x64x32xf16, #NWHC, @DDR>

// CHECK-LABEL: @ActShaveProfilingMultitile
module @ActShaveProfilingMultitile {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x128x64x32xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x128x64x32xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg5: !type_DDR) -> !type_DDR {
        %0 = memref.alloc() : !type_CMX
        %1 = VPUIP.Copy inputs(%arg0 : !type_DDR) outputs(%0 : !type_CMX) -> !type_CMX
        %2 = memref.alloc() : !type_CMX
        %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %4 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %5 = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %6 = VPUIP.SubView %2 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_MVN inputs(%3 as %arg1: !type_CMX_subview, %5 as %arg2: !type_CMX_subview) outputs(%4 as %arg3: !type_CMX_subview, %6 as %arg4: !type_CMX_subview) on tile 0 -> (!type_CMX_subview, !type_CMX_subview){
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : !type_CMX_subview, !type_CMX_subview
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : !type_CMX_subview, !type_CMX_subview
        }
        %7 = VPUIP.ConcatView inputs(%results#0, %results#1 : !type_CMX_subview, !type_CMX_subview) outputs(%2 : !type_CMX) -> !type_CMX
        %9 = VPUIP.Copy inputs(%7 : !type_CMX) outputs(%arg5 : !type_DDR) -> !type_DDR
        return %9 : !type_DDR
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<8xui32>
    //CHECK:        func.func @main(%arg0: memref<1x128x64x32xf16, #NWHC, @DDR>, %arg1: memref<1x128x64x32xf16, #NWHC, @DDR>, %arg2: memref<8xui32>) -> (memref<1x128x64x32xf16, #NWHC, @DDR>, memref<8xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = memref.alloc() : memref<8xui32, [@CMX_NN, 0]>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [8] : memref<8xui32, [@CMX_NN, 0]> to memref<8xui32, [@CMX_NN, 0]>

    //CHECK-NEXT:   [[OP_RESULT:%.*]], [[OP_RESULT_PROF:%.*]] = VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_MVN
    //CHECK-SAME:   profiling_data([[PROF_BUF_SLOT]] : memref<8xui32, [@CMX_NN, 0]>)
    //CHECK-NEXT:   VPUIP.SW.Kernel.run
    //CHECK-NEXT:   VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView %arg2 [0] [8] : memref<8xui32> to memref<8xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView inputs([[OP_RESULT_PROF]] : memref<8xui32, [@CMX_NN, 0]>) outputs([[PROF_BUF]] : memref<8xui32, [@CMX_NN, 0]>) -> memref<8xui32, [@CMX_NN, 0]>

    //CHECK:        [[PROF_BUF_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_PROF_RES]] : memref<8xui32, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT]] : memref<8xui32>) -> memref<8xui32>
    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[PROF_BUF_COPY]] : memref<8xui32>) outputs(%arg2 : memref<8xui32>) -> memref<8xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x128x64x32xf16, #NWHC, @DDR>, memref<8xui32>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!typeCmxDistributed = !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCWH, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!type_CMX_memref = memref<1x4x512x1xf16, #NCWH, @CMX_NN>

!type_DDR  = memref<1x4x512x1xf16, #NCWH, @DDR>

// CHECK-LABEL: @ActShaveProfilingMulticluster
module @ActShaveProfilingMulticluster {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x4x512x1xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x4x512x1xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg5: !type_DDR) -> !type_DDR {

        %1 = VPURT.AllocDistributed -> !typeCmxDistributed

        %2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !type_DDR) outputs(%1 as %arg2: !type_CMX_memref) -> !typeCmxDistributed {
            %3 = VPUIP.Copy inputs(%arg1 : !type_DDR) outputs(%arg2 : !type_CMX_memref) -> !type_CMX_memref
        }

        %4 = VPURT.AllocDistributed -> !typeCmxDistributed
        %5 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: !type_CMX_memref) outputs(%4 as %arg2: !type_CMX_memref) -> !typeCmxDistributed {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: !type_CMX_memref) outputs(%arg2 as %arg4: !type_CMX_memref) on tile 0 ->!type_CMX_memref {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : !type_CMX_memref, !type_CMX_memref
            }
        }

        %6 = VPUIP.NCEClusterTiling inputs(%5 as %arg1: !type_CMX_memref) outputs(%arg5 as %arg2: !type_DDR) -> !type_DDR {
            %7 = VPUIP.Copy inputs(%arg1 : !type_CMX_memref) outputs(%arg2 : !type_DDR) -> !type_DDR
        }

        return  %6 : !type_DDR
    }
    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<8xui32>
    //CHECK:         @main(%arg0: memref<1x4x512x1xf16, #NCWH, @DDR>, %arg1: memref<1x4x512x1xf16, #NCWH, @DDR>, %arg2: memref<8xui32>) -> (memref<1x4x512x1xf16, #NCWH, @DDR>, memref<8xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<8xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [8] : !VPUIP.DistributedBuffer<8xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}> to !VPUIP.DistributedBuffer<8xui32, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:        [[NCE_RES_ACT:%[0-9]+]]:2 = VPUIP.NCEClusterTiling
    //CHECK-SAME:       outputs(
    //CHECK-SAME:       [[PROF_BUF_SLOT]] as [[ARG5:%.+]]: memref<8xui32, @CMX_NN>)
    //CHECK-NEXT:       [[OP_RESULT:%.*]], [[OP_RESULT_PROF:%.*]] = VPUIP.SW.Kernel
    //CHECK-SAME:       @VPU.SW::@builtin_MVN
    //CHECK-SAME:       profiling_data([[ARG5]] : memref<8xui32, @CMX_NN>)
    //CHECK-NEXT:           VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView %arg2 [0] [8] : memref<8xui32> to memref<8xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView
    //CHECK-SAME:       inputs([[NCE_RES_ACT]]#1 : !VPUIP.DistributedBuffer<8xui32, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK-SAME:       outputs([[PROF_BUF]] : !VPUIP.DistributedBuffer<8xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)

    //CHECK:        [[NCE_RES_COPY:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[CONCAT_PROF_RES]] as [[ARG3:%.+]]: memref<8xui32, @CMX_NN>)
    //CHECK-SAME:       outputs([[PROF_OUTPUT]] as [[ARG4:%.+]]: memref<8xui32>)
    //CHECK-NEXT:       VPUIP.Copy inputs([[ARG3]] : memref<8xui32, @CMX_NN>) outputs([[ARG4]] : memref<8xui32>) -> memref<8xui32>

    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[NCE_RES_COPY]] : memref<8xui32>) outputs(%arg2 : memref<8xui32>) -> memref<8xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x4x512x1xf16, #NCWH, @DDR>, memref<8xui32>

}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!type_CMX_Distributed = !VPUIP.DistributedBuffer<
    1x128x64x32xf16, #NWHC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_CMX_Distributed_subview = !VPUIP.DistributedBuffer<
    1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_CMX = memref<1x128x64x32xf16, #NWHC, @CMX_NN>

!type_CMX_subview = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>

!type_DDR = memref<1x128x64x32xf16, #NWHC, @DDR>

// CHECK-LABEL: @ActShaveProfilingMulticlusterMultitile
module @ActShaveProfilingMulticlusterMultitile {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x128x64x32xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x128x64x32xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg9: !type_DDR) -> !type_DDR {
        %0 = VPURT.AllocDistributed -> !type_CMX_Distributed
        %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !type_DDR) outputs(%0 as %arg2: !type_CMX) -> !type_CMX_Distributed {
            %11 = VPUIP.Copy inputs(%arg1 : !type_DDR) outputs(%arg2 : !type_CMX) -> !type_CMX
        }
        %2 = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %4 = VPURT.AllocDistributed -> !type_CMX_Distributed
        %5 = VPUIP.SubView %4 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %6 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %7:2 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: !type_CMX_subview, %2 as %arg2: !type_CMX_subview) outputs(%6 as %arg3: !type_CMX_subview, %5 as %arg4: !type_CMX_subview) -> (!type_CMX_Distributed_subview, !type_CMX_Distributed_subview) {
            %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: !type_CMX_subview, %arg2 as %arg6: !type_CMX_subview) outputs(%arg3 as %arg7: !type_CMX_subview, %arg4 as %arg8: !type_CMX_subview) on tile 0 -> (!type_CMX_subview, !type_CMX_subview){
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : !type_CMX_subview, !type_CMX_subview
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : !type_CMX_subview, !type_CMX_subview
        }
        }
        %8 = VPUIP.ConcatView inputs(%7#0, %7#1 : !type_CMX_Distributed_subview, !type_CMX_Distributed_subview) outputs(%4 : !type_CMX_Distributed) -> !type_CMX_Distributed

        %9 = VPUIP.NCEClusterTiling inputs(%8 as %arg1: !type_CMX) outputs(%arg9 as %arg2: !type_DDR) -> !type_DDR {
            %11 = VPUIP.Copy inputs(%arg1 : !type_CMX) outputs(%arg2 : !type_DDR) -> !type_DDR
        }
        return %9 : !type_DDR
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<16xui32>
    //CHECK:         @main(%arg0: memref<1x128x64x32xf16, #NWHC, @DDR>, %arg1: memref<1x128x64x32xf16, #NWHC, @DDR>, %arg2: memref<16xui32>) -> (memref<1x128x64x32xf16, #NWHC, @DDR>, memref<16xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [16] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}> to !VPUIP.DistributedBuffer<16xui32, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:        [[NCE_RES_ACT:%[0-9]+]]:3 = VPUIP.NCEClusterTiling
    //CHECK-SAME:       outputs(
    //CHECK-SAME:       [[PROF_BUF_SLOT]] as [[ARG7:%.+]]: memref<16xui32, @CMX_NN>)
    //CHECK-NEXT:       [[OP_RESULT:%.*]], [[OP_RESULT_PROF:%.*]] = VPUIP.SW.Kernel
    //CHECK-SAME:       @VPU.SW::@builtin_MVN
    //CHECK-SAME:       profiling_data([[ARG7]] : memref<16xui32, @CMX_NN>)
    //CHECK-NEXT:           VPUIP.SW.Kernel.run
    //CHECK-NEXT:           VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView %arg2 [0] [16] : memref<16xui32> to memref<16xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView
    //CHECK-SAME:       inputs([[NCE_RES_ACT]]#2 : !VPUIP.DistributedBuffer<16xui32, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK-SAME:       outputs([[PROF_BUF]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)

    //CHECK:        [[NCE_RES_COPY:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[CONCAT_PROF_RES]] as [[ARG3:%.+]]: memref<16xui32, @CMX_NN>)
    //CHECK-SAME:       outputs([[PROF_OUTPUT]] as [[ARG4:%.+]]: memref<16xui32>)
    //CHECK-NEXT:       VPUIP.Copy inputs([[ARG3]] : memref<16xui32, @CMX_NN>) outputs([[ARG4]] : memref<16xui32>) -> memref<16xui32>

    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[NCE_RES_COPY]] : memref<16xui32>) outputs(%arg2 : memref<16xui32>) -> memref<16xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x128x64x32xf16, #NWHC, @DDR>, memref<16xui32>

}
