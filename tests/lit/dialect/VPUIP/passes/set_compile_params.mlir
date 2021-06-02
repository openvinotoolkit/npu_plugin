// RUN: vpux-opt --set-compile-params="vpu-arch=VPU3700" %s | FileCheck %s

// CHECK: module @test attributes {VPUIP.arch = "VPU3700", VPUIP.compilationMode = "ReferenceSW"}
module @test {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"
// CHECK-SAME:      version : {
// CHECK-SAME:          contextStr = "VPUX Compiler"
// CHECK-SAME:          hash = ""
// CHECK-SAME:          majorV = 3
// CHECK-SAME:          minorV = 11
// CHECK-SAME:          patchV = 0
// CHECK-SAME:      }

// CHECK:       IERT.RunTimeResources
// CHECK-SAME:      availableMemory :  {
// CHECK:               IERT.MemoryResource 201326592 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:               IERT.MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK:           }
// CHECK-SAME:      usedMemory :  {
// CHECK:           }
// CHECK-SAME:      executors :  {
// CHECK:               IERT.ExecutorResource 1 of "DMA_NN"
// CHECK:               IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:               IERT.ExecutorResource 4 of "NCE_Cluster" {
// CHECK:                   IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
// CHECK:               }
// CHECK:           }

}
