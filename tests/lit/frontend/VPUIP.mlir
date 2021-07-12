// RUN: cp %s %t && flatc -b --raw-binary %vpuip_schema_file% %t && vpux-translate --import-VPUIP %basename_t.blob | FileCheck %s

{
  header: {
    version: {
      majorV: 3,
      minorV: 11,
      hash: "",
      context: "VPUX Compiler"
    },
    identifier: "Test",
    net_input: [
      {
        name: "input",
        dimensions: [
          1,
          1,
          1,
          1000
        ],
        strides: [
          2.0,
          2000.0,
          2000.0,
          2000.0,
          2.0
        ],
        data: {
          data_index: 0
        },
        locale: "ProgrammableInput",
        locale_index: [
          0
        ],
        data_dtype: "FP16",
        quant_zero: [
          0
        ],
        quant_mult: [
          1
        ],
        quant_shift: [
          0
        ],
        order: 4660
      }
    ],
    net_output: [
      {
        name: "softmax",
        dimensions: [
          1,
          1,
          1,
          1000
        ],
        strides: [
          2.0,
          2000.0,
          2000.0,
          2000.0,
          2.0
        ],
        data: {
          data_index: 0
        },
        locale: "ProgrammableOutput",
        locale_index: [
          0
        ],
        data_dtype: "FP16",
        quant_zero: [
          0
        ],
        quant_mult: [
          1
        ],
        quant_shift: [
          0
        ],
        order: 4660
      }
    ],
    task_count: 3,
    options: [

    ],
    resources: {
      processor_allocation: [
        {
          item: "LEON_RT",
          number: 1.0
        },
        {
          item: "LEON_NN",
          number: 1.0
        },
        {
          item: "UPA_SHV",
          number: 16.0
        },
        {
          item: "NN_SHV",
          number: 20.0
        },
        {
          item: "NCE_PerClusterDPU",
          number: 5.0
        },
        {
          item: "NCE_Cluster",
          number: 4.0
        }
      ],
      memory_sizes: [
        {
          item: "DDR",
          number: 2048.0
        },
        {
          item: "NN_CMX",
          number: 917504.0
        }
      ]
    },
    in_tensor_desc: [
      {
        name: "input",
        dimensions: [
          1,
          1000
        ],
        strides: [
          4.0,
          4000.0,
          4.0
        ],
        data: {
          data_index: 0
        },
        locale: "ProgrammableInput",
        locale_index: [
          0
        ],
        data_dtype: "FP32",
        quant_zero: [
          0
        ],
        quant_mult: [
          1
        ],
        quant_shift: [
          0
        ],
        order: 18
      }
    ],
    out_tensor_desc: [
      {
        name: "softmax",
        dimensions: [
          1,
          1000
        ],
        strides: [
          4.0,
          4000.0,
          4.0
        ],
        data: {
          data_index: 0
        },
        locale: "ProgrammableOutput",
        locale_index: [
          0
        ],
        data_dtype: "FP32",
        quant_zero: [
          0
        ],
        quant_mult: [
          1
        ],
        quant_shift: [
          0
        ],
        order: 18
      }
    ],
    device: "KMB",
    device_revision: "B0"
  },
  task_lists: [
    {
      content: [
        {
          nodeID: 2,
          associated_barriers: {
            wait_barriers: [
              0
            ],
            update_barriers: [

            ],
            virtual_wait_barriers: [
              0
            ],
            virtual_update_barriers: [

            ]
          },
          task_type: "NNDMATask",
          task: {
            src: {
              name: "temp-0",
              dimensions: [
                1,
                1,
                1,
                1000
              ],
              strides: [
                2.0,
                2000.0,
                2000.0,
                2000.0,
                2.0
              ],
              data: {
                data_index: 0
              },
              locale: "VPU_DDR_Heap",
              locale_index: [
                0
              ],
              data_dtype: "FP16",
              quant_zero: [
                0
              ],
              quant_mult: [
                1
              ],
              quant_shift: [
                0
              ],
              order: 4660
            },
            dst: {
              name: "softmax",
              dimensions: [
                1,
                1,
                1,
                1000
              ],
              strides: [
                2.0,
                2000.0,
                2000.0,
                2000.0,
                2.0
              ],
              data: {
                data_index: 0
              },
              locale: "ProgrammableOutput",
              locale_index: [
                0
              ],
              data_dtype: "FP16",
              quant_zero: [
                0
              ],
              quant_mult: [
                1
              ],
              quant_shift: [
                0
              ],
              order: 4660
            }
          }
        }
      ]
    },
    {
      content: [
        {
          associated_barriers: {
            wait_barriers: [

            ],
            update_barriers: [

            ],
            virtual_wait_barriers: [

            ],
            virtual_update_barriers: [

            ]
          },
          task_type: "ControllerTask",
          task: {
            task_type: "BarrierConfigurationTask",
            task: {
              target: {
                consumer_count: 1,
                producer_count: 1
              }
            }
          }
        }
      ]
    },
    {
      content: [
        {
          nodeID: 1,
          associated_barriers: {
            wait_barriers: [

            ],
            update_barriers: [
              0
            ],
            virtual_wait_barriers: [

            ],
            virtual_update_barriers: [
              0
            ]
          },
          task_type: "UPALayerTask",
          task: {
            maxShaves: 16,
            softLayerParams_type: "SoftmaxParams",
            softLayerParams: {
              axis: 3
            },
            inputs: [
              {
                name: "input",
                dimensions: [
                  1,
                  1,
                  1,
                  1000
                ],
                strides: [
                  2.0,
                  2000.0,
                  2000.0,
                  2000.0,
                  2.0
                ],
                data: {
                  data_index: 0
                },
                locale: "ProgrammableInput",
                locale_index: [
                  0
                ],
                data_dtype: "FP16",
                quant_zero: [
                  0
                ],
                quant_mult: [
                  1
                ],
                quant_shift: [
                  0
                ],
                order: 4660
              }
            ],
            outputs: [
              {
                name: "temp-0",
                dimensions: [
                  1,
                  1,
                  1,
                  1000
                ],
                strides: [
                  2.0,
                  2000.0,
                  2000.0,
                  2000.0,
                  2.0
                ],
                data: {
                  data_index: 0
                },
                locale: "VPU_DDR_Heap",
                locale_index: [
                  0
                ],
                data_dtype: "FP16",
                quant_zero: [
                  0
                ],
                quant_mult: [
                  1
                ],
                quant_shift: [
                  0
                ],
                order: 4660
              }
            ]
          }
        }
      ]
    }
  ],
  barrier_table: [

  ],
  binary_data: [

  ]
}

// CHECK:   module @Test attributes {VPUIP.arch = "VPU3400"} {

// CHECK:   IERT.MemoryResource 201326592 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:   IERT.MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK:   IERT.MemoryResource 2048 bytes of "DDR"
// CHECK:   IERT.MemoryResource 917504 bytes of "CMX_NN"
// CHECK:   IERT.ExecutorResource 1 of "DMA_NN"
// CHECK:   IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:   IERT.ExecutorResource 4 of "NCE_Cluster" {
// CHECK:   IERT.ExecutorResource 5 of "NCE_PerClusterDPU"

// CHECK:   options : "NONE"
// CHECK:   contextStr = "VPUX Compiler"
// CHECK:   hash = ""
// CHECK:   majorV = 3 : i32
// CHECK:   minorV = 11 : i32
// CHECK:   patchV = 0 : i32

// CHECK:   IE.DataInfo "input" : tensor<1x1000xf32>
// CHECK:   IE.DataInfo "softmax" : tensor<1x1000xf32>

// CHECK:   func @main(%arg0: memref<1x1x1x1000xf16, #map>, %arg1: memref<1x1x1x1000xf16, #map>) -> memref<1x1x1x1000xf16, #map> {
// CHECK:   %0 = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
// CHECK:   %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x1000xf16, #map>
// CHECK:   %2 = VPUIP.NNDMA {port = 0 : i32} inputs(%1 : memref<1x1x1x1000xf16, #map>) outputs(%arg1 : memref<1x1x1x1000xf16, #map>) waits(%0 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, #map>
// CHECK:   %3 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x1000xf16, #map>
// CHECK:   %4 = VPUIP.SoftMaxUPA {axisInd = 3 : i32} inputs(%arg0 : memref<1x1x1x1000xf16, #map>) outputs(%3 : memref<1x1x1x1000xf16, #map>) updates(%0 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, #map>
// CHECK:   return %arg1 : memref<1x1x1x1000xf16, #map>
