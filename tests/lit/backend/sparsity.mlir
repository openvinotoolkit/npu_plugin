// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>

module @Test attributes {VPU.arch = "VPUX30XX"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 1048576 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
    IE.MemoryResource 1048576 bytes of @CMX_NN
}

IE.ExecutorResource 16 of @SHAVE_UPA
IE.ExecutorResource 4 of  @NCE {
    IE.ExecutorResource 5 of @DPU
}
IE.ExecutorResource 1 of @DMA_NN

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x32x32xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x16x32x32xf16>
    }

func @main(%arg0: memref<1x16x32x32xf16, #NHWC>, %arg1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    %cst_wt = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %cst_wt_table = VPURT.DeclareBuffer "CMX_NN" [0] <512> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <768> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %cst_sm = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>
    %cst_se = VPURT.DeclareBuffer "CMX_NN" [0] <49920> -> memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareSparseBuffer %0 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>, %cst_sm : memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>, %cst_se : memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]> -> !VPURT.SparseBuffer<data=memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>, storage_element_table=memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>>
    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    VPURT.Task updates(%2 : !VPURT.Barrier) {
        %3 = VPUIP.NNDMA inputs(%arg0 : memref<1x16x32x32xf16, #NHWC>) outputs(%0 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %3 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%1 : !VPURT.SparseBuffer<data=memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>, storage_element_table=memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>>) weights(%cst_wt : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%cst_wt_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%1 : !VPURT.SparseBuffer<data=memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>, storage_element_table=memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>>) parent_output(%arg1 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {end = [31, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        DPUTask {end = [31, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
        DPUTask {end = [31, 17, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
        DPUTask {end = [31, 23, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 18, 0]}
        DPUTask {end = [31, 31, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 24, 0]}
        } PPE :  {
        }
    }
    return %arg1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
}

}

// CHECK:   identifier: "Test"

// CHECK:   task_count: 3,

// CHECK:   task_type: "NCE2Task",

// CHECK:   input_data: {
// CHECK:   name: "temp-2",
// CHECK:   dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       32,
// CHECK:       32
// CHECK:   ],
// CHECK:   strides: [
// CHECK:       2.0,
// CHECK:       32768.0,
// CHECK:       2.0,
// CHECK:       1024.0,
// CHECK:       32.0
// CHECK:   ],
// CHECK:   data: {
// CHECK:       data_index: 768,
// CHECK:       sparsity_index: 33536,
// CHECK:       storage_element_index: 49920
// CHECK:   },
// CHECK:   locale: "VPU_CMX_NN",
// CHECK:   locale_index: [
// CHECK:       0
// CHECK:   ],
// CHECK:   data_dtype: "FP16",

// CHECK:   output_data: {
// CHECK:   name: "output",
// CHECK:   dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       32,
// CHECK:       32
// CHECK:   ],
// CHECK:   strides: [
// CHECK:       2.0,
// CHECK:       32768.0,
// CHECK:       2.0,
// CHECK:       1024.0,
// CHECK:       32.0
// CHECK:   ],
// CHECK:   data: {
// CHECK:       data_index: 0
// CHECK:   },
// CHECK:   locale: "ProgrammableOutput",
// CHECK:   locale_index: [
// CHECK:       0
// CHECK:   ],
// CHECK:   data_dtype: "FP16",
