// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json
//
// This file generates a blob with hswish activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPU.arch = "MTL", VPU.compilationMode = "ReferenceHW"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

IE.ExecutorResource 1 of @DMA_NN
IE.ExecutorResource 1 of @SHAVE_UPA
IE.ExecutorResource 1 of @SHAVE_ACT
IE.ExecutorResource 1 of @NCE {
    IE.ExecutorResource 1 of @DPU
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<2x2x2xf16>
        IE.DataInfo "input2" : tensor<2x2x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "output" : tensor<2x2x2xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096, // Size in bytes for the SHAVEs in the first tile.
        4096  // Size in bytes for the SHAVEs in the second tile.
    ]

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_maximum(%input : memref<*xf16>, %input2 : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "maximum.cpp",
            VPU.kernel_entry = "maximum"
        }

    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func @main(%1: memref<1x2x2x2xf16>, %2: memref<1x2x2x2xf16>,  %3: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x2x2x2xf16, @CMX_NN>
    %in2_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1x2x2x2xf16, @CMX_NN>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1x2x2x2xf16, @CMX_NN>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x2x2x2xf16>) outputs(%in_tile0_cmx : memref<1x2x2x2xf16, @CMX_NN>) -> memref<1x2x2x2xf16, @CMX_NN>
    }

    VPURT.Task updates(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%2 : memref<1x2x2x2xf16>) outputs(%in2_tile0_cmx : memref<1x2x2x2xf16, @CMX_NN>) -> memref<1x2x2x2xf16, @CMX_NN>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0, %b1 : !VPURT.Barrier, !VPURT.Barrier) updates(%b2  : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    @VPU.SW::@builtin_maximum            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx, %in2_tile0_cmx  : memref<1x2x2x2xf16,  @CMX_NN>, memref<1x2x2x2xf16, @CMX_NN>)     // Inputs/outputs buffers for generic operation interface
                    // inputs(%in2_tile0_cmx : memref<1x2x2x2xf16, @CMX_NN>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx : memref<1x2x2x2xf16, @CMX_NN>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
        -> memref<1x2x2x2xf16, @CMX_NN> {

            ^bb0(%arg0 : memref<1x2x2x2xf16, @CMX_NN>, %arg1 : memref<1x2x2x2xf16, @CMX_NN>, %arg2 : memref<1x2x2x2xf16, @CMX_NN>):
                // Inner region, isolated from above, which holds the information about arguments mapping.
                // We can use constant scalars/arrays definitions here.

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2)
                    : memref<1x2x2x2xf16, @CMX_NN>
                    , memref<1x2x2x2xf16, @CMX_NN>
                    , memref<1x2x2x2xf16, @CMX_NN>
        }
    }


    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x2x2x2xf16, @CMX_NN>) outputs(%3 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    }
    return %3: memref<1x2x2x2xf16>

}
}

// CHECK:   identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "input",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          16.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP16",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 4660,
// CHECK:        base_ptrs: [
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          16.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data_dtype: "FP16",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 4660,
// CHECK:        base_ptrs: [
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "output",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          16.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP16",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 4660,
// CHECK:        base_ptrs: [
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:   task_count: 7,

// CHECK:   options: [
// CHECK:   ],

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "input",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP16",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "output",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          2,
// CHECK:          2
// CHECK:        ],
// CHECK:        strides: [
// CHECK:          2.0,
// CHECK:          8.0,
// CHECK:          4.0,
// CHECK:          2.0
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP16",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    device: "MTL",
// CHECK:    act_kernel_runtime: {
// CHECK:      shaveStacks: [
// CHECK:        {
// CHECK:          name: "actSHAVE0_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 2,
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE1_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 3,
// CHECK:          referenced_data_size: 4096
// CHECK:        }
// CHECK:      ],
// CHECK:      kernel: {
// CHECK:        kernelText: {
// CHECK:          name: "nnActEntry",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          data_offset: 952,
// CHECK:          referenced_data_size: 656
// CHECK:        },
// CHECK:        globalArgs: {
// CHECK:          name: "nnActEntry.data",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 1,
// CHECK:          data_offset: 0
// CHECK:        }
// CHECK:      }
// CHECK:    }

// CHECK:  task_lists: [
// CHECK:    {
// CHECK:      content: [
// CHECK:        {
// CHECK:          name: "",
// CHECK:          task_type: "ControllerTask",
// CHECK:          task: {
// CHECK:            task_type: "BarrierConfigurationTask",
// CHECK:            task: {
// CHECK:              target: {
// CHECK:                consumer_count: 1,
// CHECK:                producer_count: 1
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 1,
// CHECK:          task_type: "ControllerTask",
// CHECK:          task: {
// CHECK:            task_type: "BarrierConfigurationTask",
// CHECK:            task: {
// CHECK:              target: {
// CHECK:                barrier_id: 1,
// CHECK:                consumer_count: 1,
// CHECK:                producer_count: 1
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 2,
// CHECK:          task_type: "ControllerTask",
// CHECK:          task: {
// CHECK:            task_type: "BarrierConfigurationTask",
// CHECK:            task: {
// CHECK:              target: {
// CHECK:                barrier_id: 2,
// CHECK:                consumer_count: 1,
// CHECK:                producer_count: 1
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:      ]
// CHECK:    },
// CHECK:    {
// CHECK:      content: [
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 5,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              0,
// CHECK:              1
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              2
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              0,
// CHECK:              1
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              2
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "ActKernelTask",
// CHECK:          task: {
// CHECK:            kernel: {
// CHECK:              kernelText: {
// CHECK:                name: "builtin_maximum",
// CHECK:                locale: "GFEmbeddedKernel",
// CHECK:                locale_offset: 4,
// CHECK:                data_offset: 104,
// CHECK:                referenced_data_size: 624
// CHECK:              }
// CHECK:            },


// CHECK:   kernel_data: [
// CHECK:      ]
