// RUN: vpux-opt --init-compiler="vpu-arch=KMB compilation-mode=ReferenceSW" %s | FileCheck %s

// CHECK: module @test attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"}
module @test {

// CHECK:       IERT.RunTimeResources
// CHECK-SAME:      availableMemory :  {
// CHECK:               MemoryResource 524288000 bytes of "DDR" {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
// CHECK:               MemoryResource 917504 bytes of "CMX_NN" {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK:           }
// CHECK-SAME:      usedMemory :  {
// CHECK:           }
// CHECK-SAME:      executors :  {
// CHECK:               ExecutorResource 1 of "DMA_NN"
// CHECK:               ExecutorResource 16 of "SHAVE_UPA"
// CHECK:               ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of "NCE" {
// CHECK:                   ExecutorResource 5 of "DPU"
// CHECK:               }
// CHECK:           }

}
