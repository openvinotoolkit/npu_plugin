// RUN: vpux-opt --set-compile-params="vpu-arch=KMB" %s | FileCheck %s

// CHECK: module @test attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"}
module @test {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"
// CHECK-SAME:      version : {
// CHECK-SAME:          contextStr = "VPUX Compiler"
// CHECK-SAME:          hash = "{{.*}}"
// CHECK-SAME:          majorV = {{[0-9]+}}
// CHECK-SAME:          minorV = {{[0-9]+}}
// CHECK-SAME:          patchV = {{[0-9]+}}
// CHECK-SAME:      }

// CHECK:       IERT.RunTimeResources
// CHECK-SAME:      availableMemory :  {
// CHECK:               MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:               MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK:           }
// CHECK-SAME:      usedMemory :  {
// CHECK:           }
// CHECK-SAME:      executors :  {
// CHECK:               ExecutorResource 1 of "DMA_NN"
// CHECK:               ExecutorResource 16 of "SHAVE_UPA"
// CHECK:               ExecutorResource {VPUIP.processorFrequency = 7.000000e+02 : f64} 4 of "NCE_Cluster" {
// CHECK:                   ExecutorResource 5 of "NCE_PerClusterDPU"
// CHECK:               }
// CHECK:           }

}
