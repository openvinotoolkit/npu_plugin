// RUN: vpux-opt --split-input-file --dpu-profiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfiling
module @DpuProfiling attributes {VPU.arch = "VPUX30XX", VPU.compilationMode = "DefaultHW"}  {
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 16 of @SHAVE_UPA
  IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of @NCE  {
    IE.ExecutorResource 5 of @DPU
  }
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
                end = [59, 59, 47],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 0, 0]
            }
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = IERT.Copy inputs(%1 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = IERT.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<2xui64>
    //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<2xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
    //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() : memref<2xui64, [@CMX_NN, 0]>
    //CHECK:        [[PROF_VIEW:%.+]] = IERT.SubView [[PROF_BUF_CMX]] [0] [2] : memref<2xui64, [@CMX_NN, 0]>

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<2xui64, [@CMX_NN, 0]>)

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = IERT.SubView %arg4 [0] [2] : memref<2xui64>
    //CHECK:        [[PROF_CONCAT:%.*]] = IERT.ConcatView inputs([[NCE_RES]]#1 : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = IERT.Copy inputs([[PROF_CONCAT]] : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<2xui64>)
  
    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>
    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.*]] = IERT.Copy inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = IERT.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
  
    //CHECK:        [[PROF_RES:%.*]] = IERT.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<2xui64>) outputs(%arg4 : memref<2xui64>)
  
    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x48x60x60xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!Output_DDR = type memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = type memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = type memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = type memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingWithMulticlustering
module @DpuProfilingWithMulticlustering attributes {VPU.arch = "VPUX30XX", VPU.compilationMode = "DefaultHW"}  {
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 16 of @SHAVE_UPA
  IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of @NCE  {
    IE.ExecutorResource 5 of @DPU
  }
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
        DPUTask {cluster_id = 0 : i64, end = [59, 14, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, end = [59, 29, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 15, 0]}
        DPUTask {cluster_id = 2 : i64, end = [59, 44, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 30, 0]}
        DPUTask {cluster_id = 3 : i64, end = [59, 59, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 45, 0]}
      } PPE :  {
      }
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg4: !Output_CMX) outputs(%2 as %arg5: !Output_DDR) -> !Output_DDR {
      %5 = IERT.Copy inputs(%arg4 : !Output_CMX) outputs(%arg5 : !Output_DDR) -> !Output_DDR
    }
    %4 = IERT.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<2xui64>
    //CHECK:        func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<2xui64>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() : memref<2xui64, [@CMX_NN, 0]>
    //CHECK:        [[PROF_VIEW:%.+]] = IERT.SubView [[PROF_BUF_CMX]] [0] [2] : memref<2xui64, [@CMX_NN, 0]>

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x16x62x62xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       %arg1 as [[ARG2:%.+]]: memref<48x16x3x3xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       %arg2 as [[ARG3:%.+]]: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[OUTPUT_BUF_CMX]] as [[ARG4:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:       [[PROF_VIEW]] as [[ARG5:%.+]]: memref<2xui64, [@CMX_NN, 0]>) -> 
    //CHECK-SAME:       (!VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK-SAME:       memref<2xui64, [@CMX_NN, 0]>) {
    //CHECK:        VPUIP.NCEClusterTask
    //CHECK-SAME:   input([[ARG1]] : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weights([[ARG2]] : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weight_table([[ARG3]] : memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
    //CHECK-SAME:   outputs([[ARG4]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   profiling_data([[ARG5]] : memref<2xui64, [@CMX_NN, 0]>)

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = IERT.SubView %arg4 [0] [2] : memref<2xui64>
    //CHECK:        [[PROF_CONCAT:%.*]] = IERT.ConcatView inputs([[NCE_RES]]#1 : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = IERT.Copy inputs([[PROF_CONCAT]] : memref<2xui64, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<2xui64>)
    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>

    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[NCE_RES]]#0 as [[ARG1:%.+]]: memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] as [[ARG2:%.+]]: memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        IERT.Copy
    //CHECK-SAME:       inputs([[ARG1]] : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[ARG2]] : memref<1x48x60x60xf16, #NHWC, @DDR>)

    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = IERT.Copy inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[PROF_RES:%.*]] = IERT.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<2xui64>) outputs(%arg4 : memref<2xui64>)

    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<2xui64>
}
