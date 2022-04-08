//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dpu-profiling %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfiling
module @DpuProfiling attributes {VPU.compilationMode = "DefaultHW"}  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16, {order = #NHWC}>
    DataInfo "weights" : tensor<48x16x3x3xf16, {order = #NHWC}>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
  } profilingOutputsInfo :  {
  }

  func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = memref.alloc() : !Output_CMX
    %1 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }  input(%arg0 : !Input_CMX)
            weights(%arg1 : !Weights_CMX)
            weight_table(%arg2 : !WeightsTable_CMX)
            parent_input(%arg0 : !Input_CMX)
            parent_output(%0 : !Output_CMX)
            outputs(%0 : !Output_CMX)
            -> !Output_CMX variants :  {
            DPUTask {
                outEnd = [59, 59, 47],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                outStart = [0, 0, 0]
            }
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.Copy inputs(%1 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<2xui64>
    //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<2xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
    //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() : memref<2xui64, [@CMX_NN, 0]>
    //CHECK:        [[PROF_VIEW:%.+]] = VPUIP.SubView [[PROF_BUF_CMX]] [0] [2] : memref<2xui64, [@CMX_NN, 0]> to memref<2xui64, [@CMX_NN, 0]>

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<2xui64, [@CMX_NN, 0]>)

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [2] : memref<2xui64> to memref<2xui64>
    //CHECK:        [[PROF_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#1 : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<2xui64, [@CMX_NN, 0]>) -> memref<2xui64, [@CMX_NN, 0]>
    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.Copy inputs([[PROF_CONCAT]] : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<2xui64>)

    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>
    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.Copy inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)

    //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<2xui64>) outputs(%arg4 : memref<2xui64>)

    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x48x60x60xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingWithMulticlustering
module @DpuProfilingWithMulticlustering attributes {VPU.compilationMode = "DefaultHW"}  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16, {order = #NHWC}>
    DataInfo "weights" : tensor<48x16x3x3xf16, {order = #NHWC}>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
  } profilingOutputsInfo :  {
  }

  func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.AllocDistributed -> !OutputDistributed
    %1 = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg4: !Input_CMX,
               %arg1 as %arg5: !Weights_CMX,
               %arg2 as %arg6: !WeightsTable_CMX)
        outputs(%0 as %arg7: !Output_CMX)
            -> !OutputDistributed {
      %5 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        } input(%arg4 : !Input_CMX)
          weights(%arg5 : !Weights_CMX)
          weight_table(%arg6 : !WeightsTable_CMX)
          parent_input(%arg4 : !Input_CMX)
          parent_output(%arg7 : !Output_CMX)
          outputs(%arg7 : !Output_CMX)
              -> !Output_CMX variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [59, 14, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [59, 29, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 15, 0]}
        DPUTask {cluster_id = 2 : i64, outEnd = [59, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 30, 0]}
        DPUTask {cluster_id = 3 : i64, outEnd = [59, 59, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 45, 0]}
      } PPE :  {
      }
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg4: !Output_CMX) outputs(%2 as %arg5: !Output_DDR) -> !Output_DDR {
      %5 = VPUIP.Copy inputs(%arg4 : !Output_CMX) outputs(%arg5 : !Output_DDR) -> !Output_DDR
    }
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<8xui64>
    //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<8xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<8xui64>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK:        [[PROF_BUF_CMX:%.+]] =   VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<8xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64}>
    //CHECK:        [[PROF_BUF_VIEW_CMX:%.+]] =   VPUIP.SubView [[PROF_BUF_CMX]] [0] [8]

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x16x62x62xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       %arg1 as [[ARG2:%.+]]: memref<48x16x3x3xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       %arg2 as [[ARG3:%.+]]: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[OUTPUT_BUF_CMX]] as [[ARG4:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       [[PROF_BUF_VIEW_CMX]] as [[ARG5:%.+]]: memref<8xui64, @CMX_NN>) ->
    //CHECK-SAME:       (!VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK-SAME:        !VPUIP.DistributedBuffer<8xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64}>
    //CHECK:        VPUIP.NCEClusterTask
    //CHECK-SAME:   input([[ARG1]] : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weights([[ARG2]] : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weight_table([[ARG3]] : memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
    //CHECK-SAME:   outputs([[ARG4]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   profiling_data([[ARG5]] : memref<8xui64, @CMX_NN>)

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [8] : memref<8xui64>
    //CHECK:        [[PROF_VIEW_CMX_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#1
    //CHECK-SAME:       outputs([[PROF_BUF_CMX]]

    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[PROF_VIEW_CMX_CONCAT]] as [[ARG1:%.+]]: memref<8xui64, @CMX_NN>)
    //CHECK-SAME:       outputs([[PROF_OUTPUT_VIEW]] as [[ARG2:%.+]]: memref<8xui64>)
    //CHECK:        VPUIP.Copy
    //CHECK-SAME:       inputs([[ARG1]] : memref<8xui64, @CMX_NN>)
    //CHECK-SAME:       outputs([[ARG2]] : memref<8xui64>)

    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>

    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[NCE_RES]]#0 as [[ARG1:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] as [[ARG2:%.+]]: memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        VPUIP.Copy
    //CHECK-SAME:       inputs([[ARG1]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[ARG2]] : memref<1x48x60x60xf16, #NHWC, @DDR>)

    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<8xui64>) outputs(%arg4 : memref<8xui64>)

    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<8xui64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0_CMX = type memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
!Output0_CMX = type memref<1x48x222x222xf16, #NHWC, [@CMX_NN, 0]>
!Weights0_CMX = type memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable0_CMX = type memref<48x1x1x4xsi32, [@CMX_NN, 0]>

!Weights1_CMX = type memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable1_CMX = type memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!Output1_CMX = type memref<1x32x55x55xf16, #NHWC, [@CMX_NN, 0]>
!Output2_CMX = type memref<1x32x55x55xf16, #NHWC, @CMX_NN>

!OutputDistributed = type !VPUIP.DistributedBuffer<1x32x55x55xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64
}>

!Output_DDR = type memref<1x32x55x55xf16, #NHWC>

// CHECK-LABEL: @DpuProfilingMultipleOps
module @DpuProfilingMultipleOps attributes {VPU.compilationMode = "DefaultHW"}  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x3x224x224xf16, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x32x55x55xf16, {order = #NHWC}>
  } profilingOutputsInfo :  {
  }

  func @main(%arg0: memref<1x3x224x224xf16, #NHWC>, %arg1: !Output_DDR) -> !Output_DDR {
    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<172xui64>
    //CHECK:        func @main(%arg0: memref<1x3x224x224xf16, #NHWC>, %arg1: memref<1x32x55x55xf16, #NHWC>, %arg2: memref<172xui64>) -> (memref<1x32x55x55xf16, #NHWC>, memref<172xui64>)
    //CHECK:        [[BUFFER_D:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<42xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK:        [[BUFFER_1:%.+]] = memref.alloc() : memref<10xui64, [@CMX_NN, 0]>
    //CHECK:        [[BUFFER_0:%.+]] = memref.alloc() : memref<120xui64, [@CMX_NN, 0]>

    %0 = memref.alloc() : !Input0_CMX
    %1 = memref.alloc() : !Output0_CMX
    %2 = memref.alloc() : !Weights0_CMX
    %3 = memref.alloc() : !WeightsTable0_CMX
    %4 = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          kernel_size = [3, 3],
          kernel_strides = [1, 1],
          task_type = "CONV"
        } input(%0 : !Input0_CMX)
          weights(%2 : !Weights0_CMX)
          weight_table(%3 : !WeightsTable0_CMX)
          parent_input(%0 : !Input0_CMX)
          parent_output(%1 : !Output0_CMX)
          outputs(%1 : !Output0_CMX)
          -> !Output0_CMX variants :
        {
          DPUTask {outEnd = [111, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
          DPUTask {outEnd = [221, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [112, 0, 0]}
          DPUTask {outEnd = [111, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 45, 0]}
          DPUTask {outEnd = [221, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [112, 45, 0]}
          DPUTask {outEnd = [111, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 90, 0]}
          DPUTask {outEnd = [221, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [112, 90, 0]}
          DPUTask {outEnd = [111, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 135, 0]}
          DPUTask {outEnd = [221, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [112, 135, 0]}
          DPUTask {outEnd = [111, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 180, 0]}
          DPUTask {outEnd = [221, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [112, 180, 0]}
        } PPE :  {
    }
    //CHECK:        [[PROF_VIEW_OP_0:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [0] [20] : memref<120xui64, [@CMX_NN, 0]> to memref<20xui64, [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_0:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_0]] : memref<20xui64, [@CMX_NN, 0]>)

    %5 = memref.alloc() : !Output0_CMX
    %6 = VPUIP.NCEClusterTask {
          activation_window_channel_length = 0 : i64,
          task_type = "ELTWISE"
          } input(%4 : !Output0_CMX)
            weights(%4 : !Output0_CMX)
            parent_input(%4 : !Output0_CMX)
            parent_output(%5 : !Output0_CMX)
            outputs(%5 : !Output0_CMX)
            -> !Output0_CMX variants :
          {
            DPUTask {outEnd = [31, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
            DPUTask {outEnd = [63, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [32, 0, 0]}
            DPUTask {outEnd = [95, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [64, 0, 0]}
            DPUTask {outEnd = [127, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [96, 0, 0]}
            DPUTask {outEnd = [159, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [128, 0, 0]}
            DPUTask {outEnd = [191, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [160, 0, 0]}
            DPUTask {outEnd = [221, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [192, 0, 0]}
            DPUTask {outEnd = [31, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 45, 0]}
            DPUTask {outEnd = [63, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [32, 45, 0]}
            DPUTask {outEnd = [95, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [64, 45, 0]}
            DPUTask {outEnd = [127, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [96, 45, 0]}
            DPUTask {outEnd = [159, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [128, 45, 0]}
            DPUTask {outEnd = [191, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [160, 45, 0]}
            DPUTask {outEnd = [221, 89, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [192, 45, 0]}
            DPUTask {outEnd = [31, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 90, 0]}
            DPUTask {outEnd = [63, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [32, 90, 0]}
            DPUTask {outEnd = [95, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [64, 90, 0]}
            DPUTask {outEnd = [127, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [96, 90, 0]}
            DPUTask {outEnd = [159, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [128, 90, 0]}
            DPUTask {outEnd = [191, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [160, 90, 0]}
            DPUTask {outEnd = [221, 134, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [192, 90, 0]}
            DPUTask {outEnd = [31, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 135, 0]}
            DPUTask {outEnd = [63, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [32, 135, 0]}
            DPUTask {outEnd = [95, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [64, 135, 0]}
            DPUTask {outEnd = [127, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [96, 135, 0]}
            DPUTask {outEnd = [159, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [128, 135, 0]}
            DPUTask {outEnd = [191, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [160, 135, 0]}
            DPUTask {outEnd = [221, 179, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [192, 135, 0]}
            DPUTask {outEnd = [31, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 180, 0]}
            DPUTask {outEnd = [63, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [32, 180, 0]}
            DPUTask {outEnd = [95, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [64, 180, 0]}
            DPUTask {outEnd = [127, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [96, 180, 0]}
            DPUTask {outEnd = [159, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [128, 180, 0]}
            DPUTask {outEnd = [191, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [160, 180, 0]}
            DPUTask {outEnd = [221, 221, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [192, 180, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }

    //CHECK:        [[PROF_VIEW_OP_1:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [20] [70] : memref<120xui64, [@CMX_NN, 0]> to memref<70xui64, [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_1:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_1]] : memref<70xui64, [@CMX_NN, 0]>)

    %7 = memref.alloc() : !Weights1_CMX
    %8 = memref.alloc() : !WeightsTable1_CMX
    %9 = memref.alloc() : !Output1_CMX
    %10 = VPUIP.NCEClusterTask {
        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        kernel_size = [3, 3],
        kernel_strides = [4, 4],
        task_type = "CONV"
        } input(%6 : !Output0_CMX)
          weights(%7 : !Weights1_CMX)
          weight_table(%8 : !WeightsTable1_CMX)
          parent_input(%6 : !Output0_CMX)
          parent_output(%9 : !Output1_CMX)
          outputs(%9 : !Output1_CMX)
          -> !Output1_CMX variants :
        {
          DPUTask {outEnd = [11, 18, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
          DPUTask {outEnd = [23, 18, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [12, 0, 0]}
          DPUTask {outEnd = [35, 18, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [24, 0, 0]}
          DPUTask {outEnd = [47, 18, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [36, 0, 0]}
          DPUTask {outEnd = [54, 18, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [48, 0, 0]}
          DPUTask {outEnd = [11, 37, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 19, 0]}
          DPUTask {outEnd = [23, 37, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [12, 19, 0]}
          DPUTask {outEnd = [35, 37, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [24, 19, 0]}
          DPUTask {outEnd = [47, 37, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [36, 19, 0]}
          DPUTask {outEnd = [54, 37, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [48, 19, 0]}
          DPUTask {outEnd = [11, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 38, 0]}
          DPUTask {outEnd = [23, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [12, 38, 0]}
          DPUTask {outEnd = [35, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [24, 38, 0]}
          DPUTask {outEnd = [47, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [36, 38, 0]}
          DPUTask {outEnd = [54, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [48, 38, 0]}
    } PPE :  {
    }

    //CHECK:        [[PROF_VIEW_OP_2:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [90] [30] : memref<120xui64, [@CMX_NN, 0]> to memref<30xui64, [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_2:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_2]] : memref<30xui64, [@CMX_NN, 0]>)

    //CHECK:        [[DDR_VIEW_0:%.*]] = VPUIP.SubView %arg2 [0] [120] : memref<172xui64> to memref<120xui64>
    //CHECK:        [[PROF_CONCAT_0:%.*]] = VPUIP.ConcatView inputs([[OP_RESULT_0]]#1, [[OP_RESULT_1]]#1, [[OP_RESULT_2]]#1 : memref<20xui64, [@CMX_NN, 0]>, memref<70xui64, [@CMX_NN, 0]>, memref<30xui64, [@CMX_NN, 0]>) outputs([[BUFFER_0]] : memref<120xui64, [@CMX_NN, 0]>) -> memref<120xui64, [@CMX_NN, 0]>
    //CHECK:        [[COPY_PROF_TO_DDR_0:%.*]] = VPUIP.Copy inputs([[PROF_CONCAT_0]] : memref<120xui64, [@CMX_NN, 0]>) outputs([[DDR_VIEW_0]] : memref<120xui64>) -> memref<120xui64>

    %11 = memref.alloc() : !Output1_CMX

    %12 = VPUIP.NCEClusterTask {
        activation_window_channel_length = 0 : i64,
        task_type = "ELTWISE"
        }
        input(%10 : !Output1_CMX)
        weights(%10 : !Output1_CMX)
        parent_input(%10 : !Output1_CMX)
        parent_output(%11 : !Output1_CMX)
        outputs(%11 : !Output1_CMX)
        -> !Output1_CMX variants :  {
      DPUTask {outEnd = [54, 10, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
      DPUTask {outEnd = [54, 21, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 11, 0]}
      DPUTask {outEnd = [54, 32, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 22, 0]}
      DPUTask {outEnd = [54, 43, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 33, 0]}
      DPUTask {outEnd = [54, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 44, 0]}
    } PPE :  {
      PPETask "AND" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }

    //CHECK:        [[PROF_VIEW_OP_3:%.+]] = VPUIP.SubView [[BUFFER_1]] [0] [10] : memref<10xui64, [@CMX_NN, 0]> to memref<10xui64, [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_3:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_3]] : memref<10xui64, [@CMX_NN, 0]>)

    //CHECK:        [[DDR_VIEW_1:%.*]] = VPUIP.SubView %arg2 [120] [10] : memref<172xui64> to memref<10xui64>
    //CHECK:        [[PROF_CONCAT_1:%.*]] = VPUIP.ConcatView inputs([[OP_RESULT_3]]#1 : memref<10xui64, [@CMX_NN, 0]>) outputs([[BUFFER_1]] : memref<10xui64, [@CMX_NN, 0]>) -> memref<10xui64, [@CMX_NN, 0]>
    //CHECK:        [[COPY_PROF_TO_DDR_1:%.*]] = VPUIP.Copy inputs([[PROF_CONCAT_1]] : memref<10xui64, [@CMX_NN, 0]>) outputs([[DDR_VIEW_1]] : memref<10xui64>) -> memref<10xui64>

    %13 = VPURT.AllocDistributed -> !OutputDistributed
    %14 = VPUIP.NCEClusterTiling
        inputs(%12 as %arg2: !Output1_CMX)
        outputs(%13 as %arg3: !Output2_CMX)
            -> !OutputDistributed {
         %inner = VPUIP.NCEClusterTask {
          activation_window_channel_length = 0 : i64,
          task_type = "ELTWISE"
          }
          input(%arg2 : !Output1_CMX)
          weights(%arg2 : !Output1_CMX)
          parent_input(%arg2 : !Output1_CMX)
          parent_output(%arg3 : !Output2_CMX)
          outputs(%arg3 : !Output2_CMX)
          -> !Output2_CMX variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 10, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 21, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 11, 0]}
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 32, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 22, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [54, 43, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 33, 0]}
        DPUTask {cluster_id = 2 : i64, outEnd = [54, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 44, 0]}
      } PPE :  {
        PPETask "AND" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }

    //CHECK:        [[PROF_VIEW_OP_4:%.+]] = VPUIP.SubView [[BUFFER_D]] [0] [18] : !VPUIP.DistributedBuffer<42xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK-SAME:     to !VPUIP.DistributedBuffer<18xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK:        [[OP_RESULT_4:%[0-9]+]]:2 = VPUIP.NCEClusterTiling
    //CHECK-SAME:     [[PROF_VIEW_OP_4]] as [[ARG4:%.*]]: memref<18xui64, @CMX_NN>
    //CHECK:        VPUIP.NCEClusterTask
    //CHECK-SAME:     profiling_data([[ARG4]] : memref<18xui64, @CMX_NN>)

    %15 = VPURT.AllocDistributed -> !OutputDistributed
    %16 = VPUIP.NCEClusterTiling
        inputs(%14 as %arg2: !Output1_CMX)
        outputs(%15 as %arg3: !Output2_CMX)
            -> !OutputDistributed {
         %inner = VPUIP.NCEClusterTask {
          activation_window_channel_length = 0 : i64,
          task_type = "ELTWISE"
          }
          input(%arg2 : !Output1_CMX)
          weights(%arg2 : !Output1_CMX)
          parent_input(%arg2 : !Output1_CMX)
          parent_output(%arg3 : !Output2_CMX)
          outputs(%arg3 : !Output2_CMX)
          -> !Output2_CMX variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 10, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 21, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 11, 0]}
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 32, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 22, 0]}
        DPUTask {cluster_id = 0 : i64, outEnd = [54, 43, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 33, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [54, 43, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 33, 0]}
        DPUTask {cluster_id = 2 : i64, outEnd = [54, 54, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 44, 0]}
      } PPE :  {
        PPETask "AND" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }

    //CHECK:        [[PROF_VIEW_OP_5:%.+]] = VPUIP.SubView [[BUFFER_D]] [6] [24] : !VPUIP.DistributedBuffer<42xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK-SAME:     to !VPUIP.DistributedBuffer<24xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK:        [[OP_RESULT_5:%[0-9]+]]:2 = VPUIP.NCEClusterTiling
    //CHECK-SAME:     [[PROF_VIEW_OP_5]] as [[ARG4:%.*]]: memref<24xui64, @CMX_NN>
    //CHECK:        VPUIP.NCEClusterTask
    //CHECK-SAME:     profiling_data([[ARG4]] : memref<24xui64, @CMX_NN>)


    %17 = memref.alloc() : !Output_DDR
    %18 = VPUIP.NCEClusterTiling inputs(%16 as %arg3: !Output2_CMX) outputs(%17 as %arg4: !Output_DDR) -> !Output_DDR {
      %inner = VPUIP.Copy inputs(%arg3 : !Output2_CMX) outputs(%arg4 : !Output_DDR) -> !Output_DDR
    }
    %19 = VPUIP.Copy inputs(%17 : !Output_DDR) outputs(%arg1 : !Output_DDR) -> !Output_DDR


    //CHECK:        [[DDR_VIEW_2:%.*]] = VPUIP.SubView %arg2 [130] [42] : memref<172xui64> to memref<42xui64>
    //CHECK:        [[PROF_CONCAT_3:%.*]] = VPUIP.ConcatView inputs([[OP_RESULT_4]]#1, [[OP_RESULT_5]]#1
    //CHECK-SAME:     !VPUIP.DistributedBuffer<18xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK-SAME:     !VPUIP.DistributedBuffer<24xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64}>
    //CHECK:        [[COPY_PROF_TO_DDR_2:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:     inputs([[PROF_CONCAT_3]] as [[ARG5:%.*]]: memref<42xui64, @CMX_NN>)
    //CHECK-SAME:     outputs([[DDR_VIEW_2]] as [[ARG6:%.*]]: memref<42xui64>) -> memref<42xui64>
    //CHECK:        VPUIP.Copy
    //CHECK-SAME:     inputs([[ARG5]] : memref<42xui64, @CMX_NN>)
    //CHECK-SAME:     outputs([[ARG6]] : memref<42xui64>) -> memref<42xui64>

    //CHECK:        [[PROF_RESULT:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR_0]], [[COPY_PROF_TO_DDR_1]], [[COPY_PROF_TO_DDR_2]] : memref<120xui64>, memref<10xui64>, memref<42xui64>) outputs(%arg2 : memref<172xui64>) -> memref<172xui64>

    return %19: !Output_DDR

    //CHECK:        return
    //CHECK-SAME:   [[PROF_RESULT]]
    //CHECK-SAME:   memref<172xui64>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>
!Output_DDR_SM = type memref<1x48x60x60xi1, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Output_CMX_SM = type memref<1x48x60x60xi1, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparse
module @DpuProfilingSparse attributes {VPU.compilationMode = "DefaultHW"}  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16, {order = #NHWC}>
    DataInfo "weights" : tensor<48x16x3x3xf16, {order = #NHWC}>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
  } profilingOutputsInfo :  {
  }

  func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = memref.alloc() : !Output_CMX
    %sm = memref.alloc() : !Output_CMX_SM

    %1:2 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }  input(%arg0 : !Input_CMX)
            weights(%arg1 : !Weights_CMX)
            weight_table(%arg2 : !WeightsTable_CMX)
            parent_input(%arg0 : !Input_CMX)
            parent_output(%0 : !Output_CMX)
            parent_output_sparsity_map(%sm : !Output_CMX_SM)
            outputs(%0 : !Output_CMX)
            output_sparsity_map(%sm : !Output_CMX_SM)
            -> !Output_CMX, !Output_CMX_SM variants :  {
            DPUTask {
                outEnd = [59, 59, 47],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                outStart = [0, 0, 0]
            }
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %osm = memref.alloc() : !Output_DDR_SM
    %3 = VPUIP.Copy inputs(%1#0 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    %5 = VPUIP.Copy inputs(%1#1 : !Output_CMX_SM) outputs(%osm : !Output_DDR_SM) -> !Output_DDR_SM
    return %4 : !Output_DDR
  }

  //CHECK:        profilingOutputsInfo
  //CHECK-NEXT:   DataInfo "dpu" : tensor<2xui64>
  //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<2xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
  //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() : memref<2xui64, [@CMX_NN, 0]>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @CMX_NN>
  //CHECK:        [[PROF_VIEW:%.+]] = VPUIP.SubView [[PROF_BUF_CMX]] [0] [2] : memref<2xui64, [@CMX_NN, 0]> to memref<2xui64, [@CMX_NN, 0]>

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTask
  //CHECK-SAME:   output_sparsity_map([[SPARSITY_MAP_BUF_CMX]] : memref<1x48x60x60xi1, #NHWC, @CMX_NN>)
  //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<2xui64, [@CMX_NN, 0]>)

  //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [2] : memref<2xui64> to memref<2xui64>
  //CHECK:        [[PROF_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#2 : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<2xui64, [@CMX_NN, 0]>) -> memref<2xui64, [@CMX_NN, 0]>
  //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.Copy inputs([[PROF_CONCAT]] : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<2xui64>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>
  //CHECK:        [[OUTPUT_SPARSITY_MAP_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @DDR>
  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.Copy inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.*]] = VPUIP.Copy inputs([[NCE_RES]]#1 : memref<1x48x60x60xi1, #NHWC, @CMX_NN>) outputs([[OUTPUT_SPARSITY_MAP_DDR]] : memref<1x48x60x60xi1, #NHWC, @DDR>)

  //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<2xui64>) outputs(%arg4 : memref<2xui64>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x48x60x60xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributed_SM = type !VPUIP.DistributedBuffer<
    1x48x60x60xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>
!Output_DDR_SM = type memref<1x48x60x60xi1, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Output_CMX_SM = type memref<1x48x60x60xi1, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparseWithMulticlustering
module @DpuProfilingSparseWithMulticlustering attributes {VPU.compilationMode = "DefaultHW"}  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16, {order = #NHWC}>
    DataInfo "weights" : tensor<48x16x3x3xf16, {order = #NHWC}>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
  } profilingOutputsInfo :  {
  }

  func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.AllocDistributed -> !OutputDistributed
    %sm = VPURT.AllocDistributed -> !OutputDistributed_SM
    %1:2 = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg4: !Input_CMX,
               %arg1 as %arg5: !Weights_CMX,
               %arg2 as %arg6: !WeightsTable_CMX)
        outputs(%0 as %arg7: !Output_CMX,
                %sm as %arg8: !Output_CMX_SM)
            -> (!OutputDistributed, !OutputDistributed_SM) {
      %5:2 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        } input(%arg4 : !Input_CMX)
          weights(%arg5 : !Weights_CMX)
          weight_table(%arg6 : !WeightsTable_CMX)
          parent_input(%arg4 : !Input_CMX)
          parent_output(%arg7 : !Output_CMX)
          parent_output_sparsity_map(%arg8 : !Output_CMX_SM)
          outputs(%arg7 : !Output_CMX)
          output_sparsity_map(%arg8 : !Output_CMX_SM)
              -> !Output_CMX, !Output_CMX_SM variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [59, 14, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [59, 29, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 15, 0]}
        DPUTask {cluster_id = 2 : i64, outEnd = [59, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 30, 0]}
        DPUTask {cluster_id = 3 : i64, outEnd = [59, 59, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 45, 0]}
      } PPE :  {
      }
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.NCEClusterTiling inputs(%1#0 as %arg4: !Output_CMX) outputs(%2 as %arg5: !Output_DDR) -> !Output_DDR {
      %6 = VPUIP.Copy inputs(%arg4 : !Output_CMX) outputs(%arg5 : !Output_DDR) -> !Output_DDR
    }
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    %osm = memref.alloc() : !Output_DDR_SM
    %5 = VPUIP.NCEClusterTiling inputs(%1#1 as %arg4: !Output_CMX_SM) outputs(%osm as %arg5: !Output_DDR_SM) -> !Output_DDR_SM {
      %6 = VPUIP.Copy inputs(%arg4 : !Output_CMX_SM) outputs(%arg5 : !Output_DDR_SM) -> !Output_DDR_SM
    }
    return %4 : !Output_DDR
  }

  //CHECK:        profilingOutputsInfo
  //CHECK-NEXT:   DataInfo "dpu" : tensor<8xui64>
  //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<8xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<8xui64>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]]   = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK:        [[PROF_BUF_CMX:%.+]]     = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<8xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64}>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
  //CHECK:        [[PROF_BUF_VIEW_CMX:%.+]] =   VPUIP.SubView [[PROF_BUF_CMX]] [0] [8]

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTiling
  //CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x16x62x62xf16, #NHWC, @CMX_NN>,
  //CHECK-SAME:       %arg1 as [[ARG2:%.+]]: memref<48x16x3x3xf16, #NHWC, @CMX_NN>,
  //CHECK-SAME:       %arg2 as [[ARG3:%.+]]: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
  //CHECK-SAME:       outputs([[OUTPUT_BUF_CMX]] as [[ARG4:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>,
  //CHECK-SAME:       [[SPARSITY_MAP_BUF_CMX]] as [[ARG5:%.+]]: memref<1x48x60x60xi1, #NHWC, @CMX_NN>,
  //CHECK-SAME:       [[PROF_BUF_VIEW_CMX]] as [[ARG6:%.+]]: memref<8xui64, @CMX_NN>) ->
  //CHECK-SAME:       (!VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK-SAME:       !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN
  //CHECK-SAME:       !VPUIP.DistributedBuffer<8xui64, {order = #C, strides = [1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64}>

  //CHECK:        VPUIP.NCEClusterTask
  //CHECK-SAME:   input([[ARG1]] : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:   weights([[ARG2]] : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:   weight_table([[ARG3]] : memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
  //CHECK-SAME:   outputs([[ARG4]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:   output_sparsity_map([[ARG5]] : memref<1x48x60x60xi1, #NHWC, @CMX_NN>)
  //CHECK-SAME:   profiling_data([[ARG6]] : memref<8xui64, @CMX_NN>)

  //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [8] : memref<8xui64>
  //CHECK:        [[PROF_VIEW_CMX_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#2
  //CHECK-SAME:       outputs([[PROF_BUF_CMX]]

  //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NCEClusterTiling
  //CHECK-SAME:       inputs([[PROF_VIEW_CMX_CONCAT]] as [[ARG1:%.+]]: memref<8xui64, @CMX_NN>)
  //CHECK-SAME:       outputs([[PROF_OUTPUT_VIEW]] as [[ARG2:%.+]]: memref<8xui64>)
  //CHECK:              VPUIP.Copy
  //CHECK-SAME:       inputs([[ARG1]] : memref<8xui64, @CMX_NN>)
  //CHECK-SAME:       outputs([[ARG2]] : memref<8xui64>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>

  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NCEClusterTiling
  //CHECK-SAME:       inputs([[NCE_RES]]#0 as [[ARG1:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] as [[ARG2:%.+]]: memref<1x48x60x60xf16, #NHWC, @DDR>)
  //CHECK:              VPUIP.Copy
  //CHECK-SAME:       inputs([[ARG1]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:       outputs([[ARG2]] : memref<1x48x60x60xf16, #NHWC, @DDR>)

  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)

  //CHECK:        [[SPARSITY_MAP_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @DDR>
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.+]] = VPUIP.NCEClusterTiling
  //CHECK-SAME:       inputs([[NCE_RES]]#1 as [[ARG1:%.+]]: memref<1x48x60x60xi1, #NHWC, @CMX_NN>)
  //CHECK-SAME:       outputs([[SPARSITY_MAP_BUF_DDR]] as [[ARG2:%.+]]: memref<1x48x60x60xi1, #NHWC, @DDR>)
  //CHECK:              VPUIP.Copy
  //CHECK-SAME:       inputs([[ARG1]] : memref<1x48x60x60xi1, #NHWC, @CMX_NN>)
  //CHECK-SAME:       outputs([[ARG2]] : memref<1x48x60x60xi1, #NHWC, @DDR>)

  //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<8xui64>) outputs(%arg4 : memref<8xui64>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<8xui64>

}
