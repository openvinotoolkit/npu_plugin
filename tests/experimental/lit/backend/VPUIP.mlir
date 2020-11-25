// RUN: vpux-translate -export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#NC = affine_map<(d0, d1) -> (d0, d1)>

VPUIP.Graph "Test" at @main
    options : "DynamicBarriers"
    resources : {
        processor_allocation = [
            {item = "SHAVE_UPA", number = 1 : i64},
            {item = "NCE_Cluster", number = 1 : i64}
        ],
        processor_frequencies = [],
        memory_sizes = [
            {item = "DDR", number = 2048 : i64},
            {item = "CMX_NN", number = 1048576 : i64}
        ],
        memory_bandwidth = []
    }
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }
    inputsInfo : {
        VPUIP.TensorInfo "input", f32, #NC
    }
    outputsInfo : {
        VPUIP.TensorInfo "softmax", f32, #NC
    }

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" {dataIndex = 0 : i64} -> memref<1x1000xf16>
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
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2.0
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
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2.0
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

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
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
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
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
