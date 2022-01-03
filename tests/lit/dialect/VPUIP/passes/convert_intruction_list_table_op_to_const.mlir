// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --convert-itable-op-to-constant %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @Conv2dTest(%arg0: memref<1x8x20x20xf16, #NHWC>, %arg1: memref<1x16x19x19xf16, #NHWC>) -> memref<1x16x19x19xf16, #NHWC> {
    %0 = const.Declare memref<1x16x1x1xf16> =
        #const.Content<dense<2.0> : tensor<1x16x1x1xf16>>
    %1 = const.Declare memref<16x8x2x2xf16, #NHWC> =
        #const.Content<dense<2.0> : tensor<16x8x2x2xf16>, [#const.Reorder<#NHWC>]>
    %w0 = const.Declare memref<16x1x1x4xsi32> =
        #const.Content<dense<2> : tensor<16x1x1x4xsi32>>

    %2 = IERT.StaticAlloc<0> -> memref<1x16x19x19xf16, @DDR>
    %3 = IERT.StaticAlloc<11584> -> memref<1x16x19x19xf16, #NHWC, @DDR>
    %4 = IERT.StaticAlloc<0> -> memref<1x8x20x20xf16, #NHWC, @CMX_NN>

    %5 = IERT.Copy inputs(%arg0 : memref<1x8x20x20xf16, #NHWC>)
        outputs(%4 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>)
        -> memref<1x8x20x20xf16, #NHWC, @CMX_NN>

    %6 = IERT.StaticAlloc<6400> -> memref<16x8x2x2xf16, #NHWC, @CMX_NN>

    %7 = IERT.Copy inputs(%1 : memref<16x8x2x2xf16, #NHWC>)
        outputs(%6 : memref<16x8x2x2xf16, #NHWC, @CMX_NN>)
        -> memref<16x8x2x2xf16, #NHWC, @CMX_NN>

    %8 = IERT.StaticAlloc<7680> -> memref<1x16x19x19xf16, #NHWC, @CMX_NN>

    %10 = IERT.StaticAlloc<7424> -> memref<16x1x1x4xsi32, @CMX_NN>

    %11 = IERT.Copy inputs(%w0 : memref<16x1x1x4xsi32>)
        outputs(%10 : memref<16x1x1x4xsi32, @CMX_NN>)
        -> memref<16x1x1x4xsi32, @CMX_NN>
    
    %12 = VPUIP.InstructionListTableOp {
        bias = [-119, 44, -43, -31, -19, 18, 10, 0], 
        range = [-128, -109, -90, -72, -54, -36, -18, 0, 128], 
        shift = [1, -1, 0, 0, 0, -1, -1, -4]
        } 
        -> memref<32x1x1x1xsi32>

    %13 = IERT.StaticAlloc<16640> -> memref<32x1x1x1xsi32, @CMX_NN>

    %14 = IERT.Copy inputs(%12 : memref<32x1x1x1xsi32>)
        outputs(%13 : memref<32x1x1x1xsi32, @CMX_NN>)
        -> memref<32x1x1x1xsi32, @CMX_NN>

    %15 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [2, 2],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%5 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>)
        weights(%7 : memref<16x8x2x2xf16, #NHWC, @CMX_NN>)
        weight_table(%11 : memref<16x1x1x4xsi32, @CMX_NN>)
        instruction_list_table (%14 : memref<32x1x1x1xsi32, @CMX_NN>)
        parent_input(%5 : memref<1x8x20x20xf16, #NHWC, @CMX_NN>)
        parent_output(%8 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
        outputs(%8 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
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

    %16 = IERT.Copy inputs(%15 : memref<1x16x19x19xf16, #NHWC, @CMX_NN>)
        outputs(%3 : memref<1x16x19x19xf16, #NHWC, @DDR>)
        -> memref<1x16x19x19xf16, #NHWC, @DDR>

    %17 = IERT.Copy inputs(%16 : memref<1x16x19x19xf16, #NHWC, @DDR>)
        outputs(%arg1 : memref<1x16x19x19xf16, #NHWC>)
        -> memref<1x16x19x19xf16, #NHWC>
    return %17 : memref<1x16x19x19xf16, #NHWC>
}

// CHECK:       [[INSTRUCTION_LIST_TABLE:%.+]] = const.Declare memref<32x1x1x1xsi32> =
// CHECK-SAME:      #const.Content<dense<{{.*}}> : tensor<32x1x1x1xsi32>>

// CHECK:       [[INSTRUCTION_LIST_TABLE_CMX_BUF:%.+]] = IERT.StaticAlloc<16640> -> memref<32x1x1x1xsi32, @CMX_NN>
// CHECK:       [[INSTRUCTION_LIST_TABLE_CMX:%.+]] = IERT.Copy inputs([[INSTRUCTION_LIST_TABLE]] : memref<32x1x1x1xsi32>)
// CHECK-SAME:      outputs([[INSTRUCTION_LIST_TABLE_CMX_BUF]] : memref<32x1x1x1xsi32, @CMX_NN>)

// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      instruction_list_table([[INSTRUCTION_LIST_TABLE_CMX]] : memref<32x1x1x1xsi32, @CMX_NN>)

// -----
