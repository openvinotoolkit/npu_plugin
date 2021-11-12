// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB compilation-mode=DefaultHW" --convert-to-nce-ops %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Conv2dTest
func @Conv2dTest(%arg0: memref<1x16x16x16xf16, #NHWC>, %arg1: memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC> {
    %0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x1x1xf16>>

    %2 = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

    %3 = IERT.Convolution {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        }
        inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>, %0 : memref<16x16x1x1xf16, #NHWC>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>

    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16, #NHWC>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>
    return %4 : memref<1x16x16x16xf16, #NHWC>
}

// CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<{{.*}}> : tensor<1x16x1x1xf16>>
// CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare memref<16x16x1x1xf16, #NHWC>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x16x1x1xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      op_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      bias([[BIAS_CST]] : memref<1x16x1x1xf16>)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {end = [15, 2, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
// CHECK:               DPUTask {end = [15, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0]}
// CHECK:               DPUTask {end = [15, 8, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
// CHECK:               DPUTask {end = [15, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 9, 0]}
// CHECK:               DPUTask {end = [15, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x16x16xf16, #NHWC>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolTest
func @MaxPoolTest(%arg0: memref<1x16x1x4xf16, #NHWC>, %arg1: memref<1x16x1x4xf16, #NHWC>) -> memref<1x16x1x4xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x16x1x4xf16, #NHWC>

    %1 = IERT.MaxPool {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        }
        inputs(%arg0 : memref<1x16x1x4xf16, #NHWC>)
        outputs(%0 : memref<1x16x1x4xf16, #NHWC>) -> memref<1x16x1x4xf16, #NHWC>

    %2 = IERT.Copy inputs(%1 : memref<1x16x1x4xf16, #NHWC>) outputs(%arg1 : memref<1x16x1x4xf16, #NHWC>) -> memref<1x16x1x4xf16, #NHWC>
    return %2 : memref<1x16x1x4xf16, #NHWC>
}

// CHECK-DAG:   [[ACT_WINDOW_CST:%.+]] = const.Declare memref<16x1x1x16xui8>
// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x1x4xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)

// CHECK:       [[ACT_WINDOW_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x16xui8, "CMX_NN">
// CHECK:       [[ACT_WINDOW_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[ACT_WINDOW_CST]] : memref<16x1x1x16xui8>)
// CHECK-SAME:      outputs([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 4
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "MAXPOOL"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX]] : memref<16x1x1x16xui8, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {
// CHECK-SAME:              end = [3, 0, 15]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK-SAME:              start = [0, 0, 0]

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x1x4xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x1x4xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x1x4xf16, #NHWC>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddTest
func @EltwiseAddTest(%arg0: memref<1x64x28x28xf16, #NHWC>,
                     %arg1: memref<1x64x28x28xf16, #NHWC>,
                     %arg2: memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x64x28x28xf16, #NHWC>

    %1 = IERT.Add
        inputs(%arg0 : memref<1x64x28x28xf16, #NHWC>, %arg1 : memref<1x64x28x28xf16, #NHWC>)
        outputs(%0 : memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC>

    %2 = IERT.Copy inputs(%1 : memref<1x64x28x28xf16, #NHWC>) outputs(%arg2 : memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC>
    return %2 : memref<1x64x28x28xf16, #NHWC>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC>

// CHECK:       [[INPUT1_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT1_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x64x28x28xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT1_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)

// CHECK:       [[INPUT2_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT2_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg1 : memref<1x64x28x28xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT2_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 0
// CHECK-SAME:          task_type = "ELTWISE"
// CHECK-SAME:      input([[INPUT1_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[INPUT2_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT1_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {
// CHECK-SAME:              end = [27, 4, 63]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK-SAME:              start = [0, 0, 0]

// CHECK:           } PPE : {
// CHECK:               PPETask "ADD"
// CHECK:           }

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x64x28x28xf16, #NHWC>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = type !quant.uniform<u8:f16, 0.12503063725490196:128>
!qElemType2 = type !quant.uniform<u8:f16, 0.067708337073232608:128>

// CHECK-LABEL: @QuantEltwiseMulTest
func @QuantEltwiseMulTest(%arg0: memref<1x64x28x28x!qElemType0, #NHWC>,
                     %arg1: memref<1x64x28x28x!qElemType1, #NHWC>,
                     %arg2: memref<1x64x28x28x!qElemType2, #NHWC>) -> memref<1x64x28x28x!qElemType2, #NHWC> {
    %0 = memref.alloc() : memref<1x64x28x28x!qElemType2, #NHWC>

    %1 = IERT.Multiply
        inputs(%arg0 : memref<1x64x28x28x!qElemType0, #NHWC>, %arg1 : memref<1x64x28x28x!qElemType1, #NHWC>)
        outputs(%0 : memref<1x64x28x28x!qElemType2, #NHWC>) -> memref<1x64x28x28x!qElemType2, #NHWC>

    %2 = IERT.Copy inputs(%1 : memref<1x64x28x28x!qElemType2, #NHWC>) outputs(%arg2 : memref<1x64x28x28x!qElemType2, #NHWC>) -> memref<1x64x28x28x!qElemType2, #NHWC>
    return %2 : memref<1x64x28x28x!qElemType2, #NHWC>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28x!qElemType2, #NHWC>

// CHECK:       [[INPUT1_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28x!qElemType0, #NHWC, "CMX_NN">
// CHECK:       [[INPUT1_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x64x28x28x!qElemType0, #NHWC>)
// CHECK-SAME:      outputs([[INPUT1_CMX_BUF]] : memref<1x64x28x28x!qElemType0, #NHWC, "CMX_NN">)

// CHECK:       [[INPUT2_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28x!qElemType1, #NHWC, "CMX_NN">
// CHECK:       [[INPUT2_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg1 : memref<1x64x28x28x!qElemType1, #NHWC>)
// CHECK-SAME:      outputs([[INPUT2_CMX_BUF]] : memref<1x64x28x28x!qElemType1, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28x!qElemType2, #NHWC, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 0
// CHECK-SAME:          task_type = "ELTWISE"
// CHECK-SAME:      input([[INPUT1_CMX]] : memref<1x64x28x28x!qElemType0, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[INPUT2_CMX]] : memref<1x64x28x28x!qElemType1, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT1_CMX]] : memref<1x64x28x28x!qElemType0, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x64x28x28x!qElemType2, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x64x28x28x!qElemType2, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {
// CHECK-SAME:              end = [27, 4, 63]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK-SAME:              start = [0, 0, 0]

// CHECK:           } PPE : {
// CHECK:               PPETask "MULT"
// CHECK:           }

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x64x28x28x!qElemType2, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x64x28x28x!qElemType2, #NHWC>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAndSameInputsTest
func @EltwiseAndSameInputsTest(%arg0: memref<1x64x28x28xf16, #NHWC>,
                               %arg2: memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x64x28x28xf16, #NHWC>

    %1 = IERT.And
        inputs(%arg0 : memref<1x64x28x28xf16, #NHWC>, %arg0 : memref<1x64x28x28xf16, #NHWC>)
        outputs(%0 : memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC>

    %2 = IERT.Copy inputs(%1 : memref<1x64x28x28xf16, #NHWC>) outputs(%arg2 : memref<1x64x28x28xf16, #NHWC>) -> memref<1x64x28x28xf16, #NHWC>
    return %2 : memref<1x64x28x28xf16, #NHWC>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC>

// CHECK:       [[INPUT1_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT1_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x64x28x28xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT1_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 0
// CHECK-SAME:          task_type = "ELTWISE"
// CHECK-SAME:      input([[INPUT1_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[INPUT1_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT1_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {
// CHECK-SAME:              end = [27, 4, 63]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK-SAME:              start = [0, 0, 0]

// CHECK:           } PPE : {
// CHECK:               PPETask "AND"
// CHECK:           }

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x64x28x28xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x64x28x28xf16, #NHWC>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthwiseConvTest
func @DepthwiseConvTest(%arg0: memref<1x16x40x80xf16, #NHWC>,
                        %arg1: memref<1x16x37x73xf16, #NHWC>) -> memref<1x16x37x73xf16, #NHWC> {
    %0 = const.Declare memref<16x1x4x8xf16, #NHWC> =
        #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x1x1xf16>>

    %2 = memref.alloc() : memref<1x16x37x73xf16, #NHWC>

    %3 = IERT.GroupConvolution {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        }
        inputs(%arg0 : memref<1x16x40x80xf16, #NHWC>, %0 : memref<16x1x4x8xf16, #NHWC>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x37x73xf16, #NHWC>) -> memref<1x16x37x73xf16, #NHWC>

    %4 = IERT.Copy inputs(%3 : memref<1x16x37x73xf16, #NHWC>)
                   outputs(%arg1 : memref<1x16x37x73xf16, #NHWC>) -> memref<1x16x37x73xf16, #NHWC>
    return %4 : memref<1x16x37x73xf16, #NHWC>
}

// CHECK-DAG:   [[ACT_WINDOW_CST:%.+]] = const.Declare memref<16x1x1x16xui8>
// CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare memref<16x1x4x8xf16, #NHWC>
// CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<{{.*}}> : tensor<1x16x1x1xf16>>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x37x73xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x40x80xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x40x80xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x40x80xf16, #NHWC, "CMX_NN">)

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x4x8xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x1x4x8xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x1x4x8xf16, #NHWC, "CMX_NN">)

// CHECK:       [[ACT_WINDOW_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x16xui8, "CMX_NN">
// CHECK:       [[ACT_WINDOW_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[ACT_WINDOW_CST]] : memref<16x1x1x16xui8>)
// CHECK-SAME:      outputs([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x37x73xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x40x80xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX_BUF]] : memref<16x1x4x8xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      bias([[BIAS_CST]] : memref<1x16x1x1xf16>)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [4, 8]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "DWCONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x40x80xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x1x4x8xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX]] : memref<16x1x1x16xui8, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x40x80xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x37x73xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x37x73xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {end = [72, 6, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x37x73xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x37x73xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x37x73xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x37x73xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x37x73xf16, #NHWC>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Conv2dReLUTest
func @Conv2dReLUTest(%arg0: memref<1x16x16x16xf16, #NHWC>, %arg1: memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC> {
    %0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x1x1xf16>>

    %2 = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

    %3 = IERT.Convolution {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = {attrs = {}, name = "IE.ReLU"}
        }
        inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>, %0 : memref<16x16x1x1xf16, #NHWC>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>

    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16, #NHWC>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>
    return %4 : memref<1x16x16x16xf16, #NHWC>
}

// CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<{{.*}}> : tensor<1x16x1x1xf16>>
// CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare memref<16x16x1x1xf16, #NHWC>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x16x1x1xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      op_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      bias([[BIAS_CST]] : memref<1x16x1x1xf16>)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {end = [15, 2, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
// CHECK:               DPUTask {end = [15, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0]}
// CHECK:               DPUTask {end = [15, 8, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
// CHECK:               DPUTask {end = [15, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 9, 0]}
// CHECK:               DPUTask {end = [15, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
// CHECK:               PPETask "LRELU" {clamp_high = 2147483647 : i32, clamp_low = 0 : i32, lrelu_mult = 1 : i32, lrelu_shift = 0 : ui32}


// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x16x16xf16, #NHWC>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Conv2dClampTest
func @Conv2dClampTest(%arg0: memref<1x16x16x16xf16, #NHWC>, %arg1: memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC> {
    %0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x1x1xf16>>

    %2 = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

    %3 = IERT.Convolution {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = {attrs = {max = 6.0, min = 0.0}, name = "IE.Clamp"}
        }
        inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>, %0 : memref<16x16x1x1xf16, #NHWC>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>

    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16, #NHWC>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>
    return %4 : memref<1x16x16x16xf16, #NHWC>
}

// CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<{{.*}}> : tensor<1x16x1x1xf16>>
// CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare memref<16x16x1x1xf16, #NHWC>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x16x1x1xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      op_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      bias([[BIAS_CST]] : memref<1x16x1x1xf16>)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {end = [15, 2, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
// CHECK:               DPUTask {end = [15, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0]}
// CHECK:               DPUTask {end = [15, 8, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
// CHECK:               DPUTask {end = [15, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 9, 0]}
// CHECK:               DPUTask {end = [15, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
// CHECK:               PPETask "NOOP" {clamp_high = 393216 : i32, clamp_low = 0 : i32, lrelu_mult = 1 : i32, lrelu_shift = 0 : ui32}


// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x16x16xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x16x16xf16, #NHWC>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthwiseConvTwoInputsTest
func @DepthwiseConvTwoInputsTest(%arg0: memref<1x16x20x10xf16, #NHWC>,
                                 %arg1: memref<16x1x1x1xf16, #NHWC>,
                                 %arg2: memref<1x16x20x10xf16, #NHWC>) -> memref<1x16x20x10xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x16x20x10xf16, #NHWC>
    %1 = IERT.GroupConvolution {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        }
        inputs(%arg0 : memref<1x16x20x10xf16, #NHWC>, %arg1 : memref<16x1x1x1xf16, #NHWC>)
        outputs(%0 : memref<1x16x20x10xf16, #NHWC>) -> memref<1x16x20x10xf16, #NHWC>
    %2 = IERT.Copy
        inputs(%1 : memref<1x16x20x10xf16, #NHWC>)
        outputs(%arg2 : memref<1x16x20x10xf16, #NHWC>) -> memref<1x16x20x10xf16, #NHWC>
    return %2 : memref<1x16x20x10xf16, #NHWC>
}

// CHECK:       [[PADDING_CST:%.+]] = const.Declare memref<16x15x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<16x15x1x1xf16>>
// CHECK-DAG:   [[ACT_WINDOW_CST:%.+]] = const.Declare memref<16x1x1x16xui8>
// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x20x10xf16, #NHWC>

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x20x10xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x20x10xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)

// CHECK:       [[FILTER_NCHW:%.+]] = IERT.PermuteCast {dst_order = #NCHW, mem_perm = #NHWC}
// CHECK-SAME:      inputs(%arg1 : memref<16x1x1x1xf16, #NHWC>)

// CHECK:       [[CONCAT_ALLOC:%.+]] = memref.alloc() : memref<16x16x1x1xf16>
// CHECK:       [[FILTER_SUBVIEW:%.+]] = IERT.SubView [[CONCAT_ALLOC]] [0, 0, 0, 0] [16, 1, 1, 1]
// CHECK-SAME:      : memref<16x16x1x1xf16> to memref<16x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>
// CHECK:       [[PADDING_SUBVIEW:%.+]] = IERT.SubView [[CONCAT_ALLOC]] [0, 1, 0, 0] [16, 15, 1, 1]
// CHECK-SAME:      : memref<16x16x1x1xf16> to memref<16x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>
// CHECK:       [[FILTER_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_NCHW]] : memref<16x1x1x1xf16>)
// CHECK-SAME:      outputs([[FILTER_SUBVIEW]] : memref<16x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>)
// CHECK:       [[PADDING_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[PADDING_CST]] : memref<16x15x1x1xf16>)
// CHECK-SAME:      outputs([[PADDING_SUBVIEW]] : memref<16x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>)


// CHECK:       [[EXPANDED_FILTER:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[FILTER_COPY]], [[PADDING_COPY]]
// CHECK-SAME:        memref<16x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>
// CHECK-SAME:        memref<16x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}>
// CHECK-SAME:      outputs([[CONCAT_ALLOC]] : memref<16x16x1x1xf16>) -> memref<16x16x1x1xf16>

// CHECK:       [[FILTER_NHWC:%.+]] = IERT.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK-SAME:      inputs([[EXPANDED_FILTER]] : memref<16x16x1x1xf16>) -> memref<16x16x1x1xf16, #NHWC>

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_NHWC]] : memref<16x16x1x1xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)

// CHECK:       [[ACT_WINDOW_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x16xui8, "CMX_NN">
// CHECK:       [[ACT_WINDOW_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[ACT_WINDOW_CST]] : memref<16x1x1x16xui8>)
// CHECK-SAME:      outputs([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x20x10xf16, #NHWC, "CMX_NN">

// CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.WeightsTableOp
// CHECK-SAME:      op_input([[INPUT_CMX_BUF]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0, 0, 0, 0]
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          task_type = "DWCONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX]] : memref<16x1x1x16xui8, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               DPUTask {end = [9, 3, 15], mpe_mode = "VECTOR_FP16"
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK-SAME:          start = [0, 0, 0]

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x20x10xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x20x10xf16, #NHWC>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x20x10xf16, #NHWC>)
// CHECK-SAME:      outputs(%arg2 : memref<1x16x20x10xf16, #NHWC>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x20x10xf16, #NHWC>
