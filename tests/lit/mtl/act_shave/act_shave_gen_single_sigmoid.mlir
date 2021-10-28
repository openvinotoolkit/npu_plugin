// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json
//
// This file generates a blob with sigmoid activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"} {

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1073741824 bytes
        IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
    }
    executors : {
        IERT.ExecutorResource 1 of  "Leon_RT"
        IERT.ExecutorResource 1 of  "Leon_NN"
        IERT.ExecutorResource 1 of  "ACT_SHAVE"
        IERT.ExecutorResource 1 of  "SHAVE_NN"
        IERT.ExecutorResource 1 of  "NCE_Cluster" {
            IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
        }
        IERT.ExecutorResource 1 of "DMA_UPA"
        IERT.ExecutorResource 1 of "DMA_NN"
    }

VPUIP.Graph
    options : "NONE"
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "sigmoid" : tensor<1x1000xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_sigmoid(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "sigmoid_fp16.c",
            VPU.kernel_entry = "sigmoid_fp16"
        }
}

func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, "VPU_CMX_NN">
    %out_tile0_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, "VPU_CMX_NN">

    %b0 = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    %b1 = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier


    %4 = VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, "VPU_CMX_NN">) updates(%b0 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, "VPU_CMX_NN">

    // Genetic Kernel information for the scheduler.
    %sigmoid_krn =
        VPUIP.SW.Kernel
                    @VPU.SW::@builtin_sigmoid            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx : memref<1x1x1x1000xf16, "VPU_CMX_NN">)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx : memref<1x1x1x1000xf16, "VPU_CMX_NN">)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
                    waits(%b0  : !VPUIP.Barrier)
                    updates(%b1  : !VPUIP.Barrier)
        -> memref<1x1x1x1000xf16, "VPU_CMX_NN"> {

            ^bb0(%arg0 : memref<1x1x1x1000xf16, "VPU_CMX_NN">, %arg1 : memref<1x1x1x1000xf16, "VPU_CMX_NN">):
                // Inner region, isolated from above, which holds the information about arguments mapping.
                // We can use constant scalars/arrays definitions here.
                %axis   = arith.constant 110 : i64

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %axis)
                    : memref<1x1x1x1000xf16, "VPU_CMX_NN">
                    , memref<1x1x1x1000xf16, "VPU_CMX_NN">
                    , i64
        }


    %6 = VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, "VPU_CMX_NN">) outputs(%2 : memref<1x1x1x1000xf16>) waits(%b1 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16>
    return %6: memref<1x1x1x1000xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
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
// CHECK:       name: "sigmoid",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1,
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
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

// CHECK:   task_count: 5,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
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
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "sigmoid",
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
// CHECK:   ]


// CHECK:    device: "MTL",
// CHECK:    act_kernel_runtime: {
// CHECK:        shaveStacks: [
// CHECK:          {
// CHECK:            name: "actSHAVE0_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: 4096
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE1_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            locale_offset: 1,
// CHECK:            referenced_data_size: 4096
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE2_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            locale_offset: 2,
// CHECK:            referenced_data_size: 4096
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE3_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            locale_offset: 3,
// CHECK:            referenced_data_size: 4096
// CHECK:          }
// CHECK:        ],
// CHECK:        codeScratchBuffer: {
// CHECK:          name: "scratch_buffer",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 4,
// CHECK:          data_offset: 808,
// CHECK:          referenced_data_size: 65536
// CHECK:        }
// CHECK:     }

// CHECK:   task_lists: [
// CHECK:      {
// CHECK:        content: [
// CHECK:          {
// CHECK:            name: "",
// CHECK:            nodeID: 3,
// CHECK:            associated_barriers: {
// CHECK:              wait_barriers: [
// CHECK:                0
// CHECK:              ],
// CHECK:              update_barriers: [
// CHECK:                1
// CHECK:              ],
// CHECK:              virtual_wait_barriers: [
// CHECK:                0
// CHECK:              ],
// CHECK:              virtual_update_barriers: [
// CHECK:                1
// CHECK:              ]
// CHECK:            },
// CHECK:            task_type: "ActKernelTask",
// CHECK:            task: {
// CHECK:              kernel: {
// CHECK:                kernelText: {
// CHECK:                  name: "builtin_sigmoid",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  locale_offset: 5,
// CHECK:                  data_offset: 276,
// CHECK:                  referenced_data_size: 624
// CHECK:                }
// CHECK:              },
// CHECK:              invocations: [
// CHECK:                {
// CHECK:                  associatedBarriers: {
// CHECK:                    wait_barriers: [
// CHECK:                      0
// CHECK:                    ],
// CHECK:                    update_barriers: [
// CHECK:                      1
// CHECK:                    ]
// CHECK:                  },
// CHECK:                  dataSection: {
// CHECK:                    name: "builtin_sigmoid_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                    locale_offset: 6,
// CHECK:                    data_offset: 572
// CHECK:                  },
// CHECK:                  invocationArgs: {
// CHECK:                    name: "builtin_sigmoid_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                    locale_offset: 6,
// CHECK:                    data_offset: 572,
// CHECK:                    referenced_data_size: 176
// CHECK:                  }
// CHECK:                }
// CHECK:              ]
// CHECK:            }
// CHECK:          }
// CHECK:        ]
// CHECK:      },


// CHECK:   kernel_data: [
// CHECK:      {
// CHECK:        length: 4096,
// CHECK:      },
// CHECK:      {
// CHECK:        length: 4096,
// CHECK:      },
// CHECK:      {
// CHECK:        length: 4096,
// CHECK:      },
// CHECK:      {
// CHECK:        length: 4096,
// CHECK:      },
// CHECK:      {
// CHECK:        length: 66560,
// CHECK:      },
// CHECK:      ]

