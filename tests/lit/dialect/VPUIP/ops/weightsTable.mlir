// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WeightTableUseBaseBuffer
func @WeightTableUseBaseBuffer(%arg0 : memref<1x16x30x30xf16, #NHWC>) -> memref<16x1x1x4xsi32> {
    %0 = memref.alloc() : memref<1x16x30x30xf16, #NHWC, @CMX_NN>
    %1 = IERT.Copy inputs(%arg0 : memref<1x16x30x30xf16, #NHWC>) outputs(%0 : memref<1x16x30x30xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x30xf16, #NHWC, @CMX_NN>
    %2 = memref.alloc() : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
    %cst = const.Declare memref<16x1x1x16xui8> =
        #const.Content<dense<1> : tensor<16x1x1x16xui8>>
    %3 = memref.alloc() : memref<16x1x1x16xui8, @CMX_NN>
    %4 = IERT.Copy inputs(%cst : memref<16x1x1x16xui8>) outputs(%3 : memref<16x1x1x16xui8, @CMX_NN>) -> memref<16x1x1x16xui8, @CMX_NN>
    %5 = VPUIP.WeightsTableOp op_input(%1 : memref<1x16x30x30xf16, #NHWC, @CMX_NN>) op_output(%2 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>) activation_window(%4 : memref<16x1x1x16xui8, @CMX_NN>) -> memref<16x1x1x4xsi32>

    return %5 : memref<16x1x1x4xsi32>

    // CHECK:       [[ALLOC0:%.*]] = memref.alloc() : memref<1x16x30x30xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC1:%.*]] = memref.alloc() : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC2:%.*]] = memref.alloc() : memref<16x1x1x16xui8, @CMX_NN>
    // CHECK:       [[WEIGHTS:%.*]] = VPUIP.WeightsTableOp op_input([[ALLOC0]] : memref<1x16x30x30xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  op_output([[ALLOC1]] : memref<1x16x15x15xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  activation_window([[ALLOC2]] : memref<16x1x1x16xui8, @CMX_NN>)

    // CHECK:       return [[WEIGHTS]]
}
