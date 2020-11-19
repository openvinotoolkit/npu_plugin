// RUN: vpux-translate -export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#NC = affine_map<(d0, d1) -> (d0, d1)>

VPUIP.Graph "Test" at @main
    options : "DynamicBarriers"
    resources : {
        ddr_scratch = 2048 : i64,
        nn_cmx_slice_amount = 1 : i32,
        upa_shaves = 1 : i32
    }
    inputsInfo : {
        VPUIP.TensorInfo "input", f32, #NC
    }
    outputsInfo : {
        VPUIP.TensorInfo "softmax", f32, #NC
    }

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap", 0 -> memref<1x1000xf16>
    %1 = VPUIP.ConfigureBarrier -> !VPUIP.Barrier
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) updates(%1 : !VPUIP.Barrier)
    VPUIP.UPADMA inputs(%0 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) waits(%1 : !VPUIP.Barrier)
    return
}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2,
// CHECK:           2000,
// CHECK:           2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2,
// CHECK:         2000,
// CHECK:         2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 3,

// CHECK:   options: [
// CHECK:     "DynamicBarriers"
// CHECK:   ],

// CHECK:   resources: {
// CHECK:     upa_shaves: 1,
// CHECK:     nn_cmx_slice_amount: 1,
// CHECK:     ddr_scratch: 2048
// CHECK:   },

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4,
// CHECK:         4000,
// CHECK:         4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4,
// CHECK:         4000,
// CHECK:         4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ]

// CHECK:   barrier_table: [
// CHECK:     {
// CHECK:       consumer_count: 1,
// CHECK:       producer_count: 1
// CHECK:     }
// CHECK:   ],
