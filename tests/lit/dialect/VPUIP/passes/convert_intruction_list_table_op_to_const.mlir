// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --convert-itable-op-to-constant %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @Conv2dTest(%arg0: memref<1x8x20x20xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x19x19xf16, #NHWC, @CMX_NN>) -> memref<1x16x19x19xf16, #NHWC, @CMX_NN> {
    %1 = const.Declare memref<16x8x2x2xf16, #NHWC, @CMX_NN> =
        #const.Content<dense<2.0> : tensor<16x8x2x2xf16>, [#const.Reorder<#NHWC>]>
    %2 = const.Declare memref<16x1x1x4xsi32, @CMX_NN> =
        #const.Content<dense<2> : tensor<16x1x1x4xsi32>>
    
    %3 = VPUIP.InstructionListTableOp {
        bias = [-119, 44, -43, -31, -19, 18, 10, 0], 
        range = [-128, -109, -90, -72, -54, -36, -18, 0, 128], 
        shift = [1, -1, 0, 0, 0, -1, -1, -4]
        } 
        -> memref<1x1x1x32xsi32>

    %4 = IERT.StaticAlloc<16640> -> memref<1x1x1x32xsi32, @CMX_NN>

    %5 = IERT.Copy inputs(%3 : memref<1x1x1x32xsi32>)
        outputs(%4 : memref<1x1x1x32xsi32, @CMX_NN>)
        -> memref<1x1x1x32xsi32, @CMX_NN>

    %6 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [2, 2],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%arg0 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>)
        weights(%1 : memref<16x8x2x2xf16, #NHWC, @CMX_NN>)
        weight_table(%2 : memref<16x1x1x4xsi32, @CMX_NN>)
        instruction_list_table (%5 : memref<1x1x1x32xsi32, @CMX_NN>)
        parent_input(%arg0 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>)
        parent_output(%arg1 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
        -> memref<1x16x19x19xf16, #NHWC, @CMX_NN>
        variants : {
            DPUTask {
                end = [18, 2, 15],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 0, 0]
            }
            DPUTask {
                end = [18, 5, 15],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 3, 0]
            }
            DPUTask {
                end = [18, 8, 15],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 6, 0]
            }
            DPUTask {
                end = [18, 11, 15],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 9, 0]
            }
            DPUTask {
                end = [18, 18, 15],
                mpe_mode = "VECTOR_FP16",
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                start = [0, 12, 0]
            }
        } PPE : {
        }

    return %6 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>
}

// CHECK:       [[INSTRUCTION_LIST_TABLE:%.+]] = const.Declare memref<1x1x1x32xsi32> =
// CHECK-SAME:      #const.Content<dense<{{.*}}> : tensor<1x1x1x32xsi32>>

// CHECK:       [[INSTRUCTION_LIST_TABLE_CMX_BUF:%.+]] = IERT.StaticAlloc<16640> -> memref<1x1x1x32xsi32, @CMX_NN>
// CHECK:       [[INSTRUCTION_LIST_TABLE_CMX:%.+]] = IERT.Copy inputs([[INSTRUCTION_LIST_TABLE]] : memref<1x1x1x32xsi32>)
// CHECK-SAME:      outputs([[INSTRUCTION_LIST_TABLE_CMX_BUF]] : memref<1x1x1x32xsi32, @CMX_NN>)

// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      instruction_list_table([[INSTRUCTION_LIST_TABLE_CMX]] : memref<1x1x1x32xsi32, @CMX_NN>)

// -----
