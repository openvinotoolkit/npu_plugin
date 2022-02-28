// RUN: vpux-opt --split-input-file --optimize-copies %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @CopyWithSubViewOp(%in : memref<1x16x113x113xf16, #NHWC, @DDR>, 
                        %weight_table : memref<16x1x1x4xsi32, @CMX_NN>, 
                        %act_wind : memref<16x1x1x16xui8, @CMX_NN>) 
                        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN> {

    %buf0 = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>

    // activation copy-in
    %0 = IERT.Copy
            inputs(%in : memref<1x16x113x113xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x113x113xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask 
        {
            activation_window_channel_length = 27 : i64, 
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            kernel_size = [3, 3], 
            kernel_strides = [2, 2], 
            task_type = "MAXPOOL"
        } 
        input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>) 
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>) 
        activation_window(%act_wind : memref<16x1x1x16xui8, @CMX_NN>) 
        parent_input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>) 
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN> 
        variants :  
        {
            DPUTask 
                {
                    end = [55, 10, 15], mpe_mode = "VECTOR_FP16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    start = [0, 0, 0]
                }
        }   
        PPE :  
        {
        }
    
    // slice of buffer where the NCE writes
    %2 = IERT.SubView %buf2 [0, 0, 0, 0] [1, 16, 56, 56] : 
        memref<1x32x56x56xf16, #NHWC, @CMX_NN> to 
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // copy of the output NCE from NNCMX->NNCMX
    %3 = IERT.Copy 
        inputs(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
        outputs(%2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>) 
        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    return %2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // verify that the SubView operation is not removed along with the copy operation

    // CHECK:       [[VAL0:%.+]] = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAL1:%.+]] = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAL2:%.+]] = IERT.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x16x113x113xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
    // CHECK:       [[VAL3:%.+]] = IERT.SubView [[VAL1]]
    // CHECK:       [[VAL4:%.+]] = VPUIP.NCEClusterTask
    
    // copy optimized
    // CHECK-NOT:   IE.Copy

    // CHECK:       return [[VAL3:%.+]] : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

}
