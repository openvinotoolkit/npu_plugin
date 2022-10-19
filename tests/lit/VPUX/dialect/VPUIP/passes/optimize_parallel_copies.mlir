// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeParallelConstCopies(
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>
    %0 = const.Declare memref<1x16x112x112xf16, #NHWC, @DDR> = #const.Content<dense<1.000000e+00> : tensor<1x16x112x112xf16>, [#const.Reorder<#NHWC>]>
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %6 = VPUIP.Copy
            inputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %8 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %10 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %11 = VPUIP.Copy
            inputs(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

}

// CHECK-LABEL: func @OptimizeParallelConstCopies

// CHECK:       [[VAR0:%.*]] =  const.Declare memref<1x16x112x112xf16, #NHWC, @DDR>
// CHECK:       [[VAR1:%.*]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR3:%.*]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK:       [[VAR4:%.*]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR4]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR6:%.*]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SkipInputSubViewPatternCopies(
        %input: memref<1x16x112x113xf16, #NHWC, @DDR>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>
    %0 = memref.alloc() : memref<1x16x112x113xf16, #NHWC, @DDR>

    %2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.Copy
            inputs(%6 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %12 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %13 = VPUIP.Copy
            inputs(%12 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>
}

// CHECK-LABEL: func @SkipInputSubViewPatternCopies

// CHECK:       %[[ALLOC_DDR:.*]] = memref.alloc() : memref<1x16x112x113xf16, #NHWC, @DDR>
// CHECK:       %[[ALLOC_CMX_1:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[SUBVIEW_1:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       %[[DDR2CMX_1:.*]] = VPUIP.Copy inputs(%[[SUBVIEW_1]] :
// CHECK-SAME:        memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[ALLOC_NCE_CMX:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_1:.*]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_NCE_CMX:.*]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_1:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       %[[ALLOC_CMX_2:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[SUBVIEW_2:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       %[[DDR2CMX_2:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[SUBVIEW_2]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[ALLOC_CMX_3:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_2:.*]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_3]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_2:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       return %[[CMX2DDR_1]], %[[CMX2DDR_2]] : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = type memref<1x32x28x28xf16, #NHWC, @DDR>
!Output_CMX = type memref<1x32x28x28xf16, #NHWC, @CMX_NN>
!Output = type memref<1x32x28x28xf16,  #NHWC>
!Weights_CMX = type memref<128x32x1x1xf16, #NHWC, @CMX_NN>
!Output_CONV = type memref<1x128x28x28xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = type memref<128x1x1x4xsi32, @CMX_NN>

!CopyOutput_Distributed = type !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!ConvOutput_Distributed = type !VPUIP.DistributedBuffer<
  1x128x28x28xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

module @OptimizeParallelMulticlusterCopies {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
  } outputsInfo :  {
    DataInfo "output" : tensor<1x128x28x28xf16>
    DataInfo "output2" : tensor<1x128x28x28xf16>
  }
  func @main() -> (!ConvOutput_Distributed, !ConvOutput_Distributed) {
    %0 = memref.alloc() : !Output_DDR
    %1 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %2 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %3 = memref.alloc() : !Output_CMX
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg0: !Output_CMX ) outputs(%0 as %arg3: !Output) -> !Output_DDR {
      %inner = VPUIP.Copy inputs(%arg0 : !Output_CMX ) outputs(%arg3 : !Output) -> !Output 
    }
    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !Output) outputs(%1 as %arg3: !Output_CMX ) -> !CopyOutput_Distributed {
      %inner = VPUIP.Copy inputs(%arg0 : !Output) outputs(%arg3 : !Output_CMX ) -> !Output_CMX 
    }
    %6 = memref.alloc() : !Output_CONV
    %7 = memref.alloc() : !Weights_CMX
    %8 = memref.alloc() : !Weights_table_CMX
    %9 = VPUIP.NCEClusterTiling
      inputs(%5 as %arg2: !Output_CMX, %7 as %arg3: !Weights_CMX, %8 as %arg4:  !Weights_table_CMX) 
      outputs(%6 as %arg5: !Output_CONV)
       -> !ConvOutput_Distributed {
      %inner = VPUIP.NCEClusterTask {
        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = "CONV"
          } input(%arg2 : !Output_CMX)
          weights(%arg3 : !Weights_CMX)
          weight_table(%arg4 : !Weights_table_CMX)
          parent_input(%arg2 : !Output_CMX) 
          parent_output(%arg5 : !Output_CONV) 
          outputs(%arg5 : !Output_CONV) 
            -> !Output_CONV variants :  {
              DPUTask {cluster_id = 0 : i64, end = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
       } PPE :  {
      }
    }
    %10 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !Output) outputs(%2 as %arg1: !Output_CMX ) -> !CopyOutput_Distributed {
      %inner = VPUIP.Copy inputs(%arg0 : !Output) outputs(%arg1 : !Output_CMX ) -> !Output_CMX  
    }  
    %12 = VPUIP.NCEClusterTiling 
      inputs(%10 as %arg2: !Output_CMX, %7 as %arg3: !Weights_CMX, %8 as %arg4:  !Weights_table_CMX) 
      outputs(%6 as %arg5: !Output_CONV)
       -> !ConvOutput_Distributed {
      %inner = VPUIP.NCEClusterTask {
        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = "CONV"
          } input(%arg2 : !Output_CMX)
          weights(%arg3 : !Weights_CMX)
          weight_table(%arg4 : !Weights_table_CMX)
          parent_input(%arg2 : !Output_CMX) 
          parent_output(%arg5 : !Output_CONV) 
          outputs(%arg5 : !Output_CONV) 
            -> !Output_CONV variants :  {
              DPUTask {cluster_id = 0 : i64, end = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
       } PPE :  {
      }
    }
    return %9, %12: !ConvOutput_Distributed, !ConvOutput_Distributed
  }
}

// CHECK-LABEL: @OptimizeParallelMulticlusterCopies

//CHECK: [[DISTR_BUFFER0:%.*]] = VPURT.AllocDistributed
//CHECK: [[COMMON_ROOT:%.*]] = VPUIP.NCEClusterTiling

//CHECK: [[BRANCH1_COPY:%.*]] = VPUIP.NCEClusterTiling 
//CHECK-SAME:  inputs([[COMMON_ROOT]] as %arg0: memref<1x32x28x28xf16, #NHWC>)
//CHECK-SAME:  outputs([[DISTR_BUFFER0]] as %arg1: memref<1x32x28x28xf16, #NHWC, @CMX_NN>)

//CHECK: [[BRANCH1_CONSUMER:%.*]] = VPUIP.NCEClusterTiling
//CHECK-SAME: inputs([[BRANCH1_COPY]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>

//CHECK-NOT: VPUIP.NCEClusterTiling
//CHECK-NOT: VPUIP.Copy 

//CHECK: [[BRANCH2_CONSUMER:%.*]] = VPUIP.NCEClusterTiling
//CHECK-SAME: inputs([[BRANCH1_COPY]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>

//CHECK:  return [[BRANCH1_CONSUMER]], [[BRANCH2_CONSUMER]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func @OptimizeParallelSubViewWithClusterTilingCopies(
        %input: memref<1x144x128x128xf16, #NHWC, @DDR>,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x144x128x128xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
        -> memref<1x144x128x128xf16, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(
                %4 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = "CONV"}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, end = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        } PPE :  {
        }
    }

    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.NCEClusterTiling
            inputs(%7 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%8 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTiling
            inputs(
                %9 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %10 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = "CONV"}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, end = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        } PPE :  {
        }
    }

    return %6, %11 : !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.*]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.*]] = IERT.Convert inputs(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>) outputs([[IN_BUFFER]] : memref<1x144x128x128xf16, #NHWC, @DDR>) -> memref<1x144x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg3: memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUFFER_1]] as %arg4: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:       [[COPY_INNER_1:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_2]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUFFER_UNUSED:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFFER_3:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_3]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>


    // CHECK:       return [[NCE_1]], [[NCE_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:                                     !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}
