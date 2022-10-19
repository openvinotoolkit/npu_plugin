// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ParsePrintClusterTiling(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>
                = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    %weight_table = const.Declare memref<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %weight_table_buff_cmx = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %weight_table_cmx = VPUIP.Copy inputs(%weight_table : memref<64x1x1x4xsi32>) outputs(%weight_table_buff_cmx : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
                    attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling
                inputs(%arg0 as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
                        %weights as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
                        %weight_table_cmx as %arg3: memref<64x1x1x4xsi32, @CMX_NN>)
                outputs(%out_buff_cmx as %arg4: memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
                    -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {

              %4 = VPUIP.NCEClusterTask {
                        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        kernel_size = [1, 1],
                        kernel_strides = [1, 1],
                        task_type = "CONV"
                    }  input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
                        weights(%arg2 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
                        weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>)
                        parent_input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
                        parent_output(%arg4 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
                        outputs(%arg4 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
                            -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> variants :  {
                        DPUTask {
                            start = [0, 0, 0], end = [31, 15, 15],
                            mpe_mode = "VECTOR_FP16",
                            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
                        }
                        } PPE :  {
                        }
        }

        async.yield %0 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    }


    %0 = async.await %r1 : !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
    return %0 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>

    //CHECK:        [[OUT_BUFF_CMX:%.*]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    //CHECK:        [[WEIGHT_TABLE:%.*]] = const.Declare memref<64x1x1x4xsi32>

    //CHECK:        [[WEIGHT_TABLE_BUFF_CMX:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    //CHECK:        [[WEIGHT_TABLE_CMX:%.*]] = VPUIP.Copy

    //CHECK:        %token, %results = async.execute -> !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:        [[VAR0:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:           inputs(%arg0 as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[WEIGHTS]] as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[WEIGHT_TABLE_CMX]] as %arg3: memref<64x1x1x4xsi32, @CMX_NN>)
    //CHECK-SAME:           outputs([[OUT_BUFF_CMX]] as %arg4: memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {
    //CHECK:        [[VAR1:%.*]] = VPUIP.NCEClusterTask
    //CHECK-SAME:           input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:           weights(%arg2 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:           weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>)
    //CHECK-SAME:           parent_input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:           parent_output(%arg4 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:           outputs(%arg4 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    //CHECK:            }
    //CHECK:            async.yield [[VAR0]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    //CHECK:        }
    //CHECK:        [[VAR2:%.*]] = async.await %results : !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
    //CHECK:        return [[VAR2]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<64x32x3x3xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x64x16x16xf16, #NHWC, @DDR>

!WeightsTableStub = type memref<64x1x1x4xsi32>
!InputStub_CMX = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, @CMX_NN>
!OutputStub_CMX = type memref<1x64x16x16xf16, #NHWC, @CMX_NN>

func @ParsePrintDistributedBuffer(%input: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
            %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
        }

        async.yield
    }

    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %t1 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
            %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
        }

        async.yield
    }

    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %t2 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling
                inputs(%input_cmx as %arg0: !InputStub_CMX,
                        %output_buff_cmx as %arg1: !OutputStub_CMX,
                        %weights_cmx as %arg2: !WeightsStub_CMX)
                outputs(%weights_table_cmx as %arg3: !WeightsTableStub_CMX)
                    -> !WeightsTableDistributed {
            %1 = const.Declare !WeightsTableStub = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
            %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%1: !WeightsTableStub) outputs(%arg3: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
        }

        async.yield
    }

    %t3 = async.execute [%t0, %t1, %t2]
                attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
            %0 = VPUIP.NCEClusterTiling
                    inputs(%input_cmx as %arg0: !InputStub_CMX,
                            %weights_cmx as %arg1: !WeightsStub_CMX,
                            %weights_table_cmx as %arg2: !WeightsTableStub_CMX)
                    outputs(%output_buff_cmx as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

                  %1 = VPUIP.NCEClusterTask {
                            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = "CONV"
                        }  input(%arg0 : !InputStub_CMX)
                            weights(%arg1 : !WeightsStub_CMX)
                            weight_table(%arg2 : !WeightsTableStub_CMX)
                            parent_input(%arg0 : !InputStub_CMX)
                            parent_output(%arg3 : !OutputStub_CMX)
                            outputs(%arg3 : !OutputStub_CMX)
                                -> !OutputStub_CMX variants :  {
                            DPUTask {
                                start = [0, 0, 0], end = [31, 15, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
                            }
                            } PPE :  {
                            }
            }

            async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%output_buff_cmx as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
            %1 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
        }

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        %cst = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              %5 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x16x16xf16, #NHWC, @DDR>)
    //CHECK-SAME:               outputs([[INPUT_CMX]] as %arg2: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:                %6 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg1 : memref<1x32x16x16xf16, #NHWC, @DDR>)
    //CHECK-SAME:                   outputs(%arg2 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    //CHECK:              }
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              %5 = VPUIP.NCEClusterTiling inputs(%cst as %arg1: memref<64x32x3x3xf16, #NHWC, @DDR>)
    //CHECK-SAME:               outputs([[WEIGHTS_CMX]] as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               -> !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:                %6 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg1 : memref<64x32x3x3xf16, #NHWC, @DDR>)
    //CHECK-SAME:                   outputs(%arg2 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    //CHECK:              }
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT_BUFF_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    //CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    //CHECK:        %token_1 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              %5 = VPUIP.NCEClusterTiling inputs(
    //CHECK-SAME:                   [[INPUT_CMX]] as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[OUTPUT_BUFF_CMX]] as %arg2: memref<1x64x16x16xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[WEIGHTS_CMX]] as %arg3: memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               outputs([[WEIGHTS_TABLE_CMX]] as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
    //CHECK-SAME:               -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:                %cst_4 = const.Declare memref<64x1x1x4xsi32>
    //CHECK:                %6 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%cst_4 : memref<64x1x1x4xsi32>) outputs(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    //CHECK:              }
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        %token_2 = async.execute [%token, %token_0, %token_1] attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
    //CHECK:              %5 = VPUIP.NCEClusterTiling inputs(
    //CHECK-SAME:                   [[INPUT_CMX]] as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[WEIGHTS_CMX]] as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:                   [[WEIGHTS_TABLE_CMX]] as %arg3: memref<64x1x1x4xsi32, @CMX_NN>)
    //CHECK-SAME:               outputs([[OUTPUT_BUFF_CMX]] as %arg4: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               -> memref<1x64x16x16xf16, #NHWC, @CMX_NN> {
    //CHECK:                %6 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>) weights(%arg2 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>) parent_input(%arg1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>) parent_output(%arg4 : memref<1x64x16x16xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x64x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x16x16xf16, #NHWC, @CMX_NN> variants :  {
    //CHECK:                  DPUTask {end = [31, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:                } PPE :  {
    //CHECK:                }
    //CHECK:              }
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x64x16x16xf16, #NHWC, @DDR>
    //CHECK:        %token_3 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              %5 = VPUIP.NCEClusterTiling inputs([[OUTPUT_BUFF_CMX]] as %arg1: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:               outputs([[OUTPUT]] as %arg2: memref<1x64x16x16xf16, #NHWC, @DDR>)
    //CHECK-SAME:               -> memref<1x64x16x16xf16, #NHWC, @DDR> {
    //CHECK:                %6 = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg1 : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:                   outputs(%arg2 : memref<1x64x16x16xf16, #NHWC, @DDR>) -> memref<1x64x16x16xf16, #NHWC, @DDR>
    //CHECK:              }
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x64x16x16xf16, #NHWC, @DDR>
}
