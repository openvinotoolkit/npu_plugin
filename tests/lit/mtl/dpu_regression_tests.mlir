// RUN: vpux-translate --split-input-file --import-HWTEST %s | FileCheck %s

// CHECK-LABEL: module @mainModule attributes {VPUIP.arch = "VPU3720", VPUIP.compilationMode = "ReferenceSW"}
// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"
// CHECK-SAME:      version : {
// CHECK-SAME:          contextStr = "VPUX Compiler"
// CHECK-SAME:      }

// CHECK:       IERT.RunTimeResources
// CHECK-SAME:      availableMemory :  {
// CHECK:               IERT.MemoryResource 1073741824 bytes
// CHECK:               IERT.MemoryResource 201326592 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:               IERT.MemoryResource 1966080 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK:           }
// CHECK-SAME:      usedMemory :  {
// CHECK:           }
// CHECK-SAME:      executors :  {
// CHECK:               IERT.ExecutorResource 1 of "Leon_RT"
// CHECK:               IERT.ExecutorResource 1 of "Leon_NN"
// CHECK:               IERT.ExecutorResource 1 of "SHAVE_NN"
// CHECK:               IERT.ExecutorResource 1 of "NCE_Cluster" {
// CHECK:                   IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
// CHECK:               }
// CHECK:               IERT.ExecutorResource 2 of "DMA_NN"
// CHECK:           }

// -----

// CHECK-LABEL: module @mainModule attributes {VPUIP.arch = "VPU3720", VPUIP.compilationMode = "ReferenceSW"}
// CHECK: VPUIP.NCEClusterTask
// CHECK-SAME: task_type = "CONV"
