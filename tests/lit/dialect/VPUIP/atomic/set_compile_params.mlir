// RUN: vpux-opt --set-compile-params="vpu-arch=MA2490" %s | FileCheck %s

// CHECK: module @test attributes {VPUIP.arch = "MA2490"}
module @test {

// CHECK: IERT.RunTimeResources availableMemory :  {
// CHECK:   IERT.MemoryResource 1073741824 bytes
// CHECK:   IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:   IERT.MemoryResource 4194304 bytes of "CMX_UPA" {VPUIP.bandwidth = 16 : i64, VPUIP.derateFactor = 8.500000e-01 : f64}
// CHECK:   IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK: } usedMemory :  {
// CHECK: } availableExecutors :  {
// CHECK:   IERT.ExecutorResource 1 of "Leon_RT"
// CHECK:   IERT.ExecutorResource 1 of "Leon_NN"
// CHECK:   IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:   IERT.ExecutorResource 20 of "SHAVE_NN"
// CHECK:   IERT.ExecutorResource 4 of "NCE_Cluster" {
// CHECK:            IERT.ExecutorResource 5 of "NCE_PerClusterDPU" {
// CHECK:            }
// CHECK:   }
// CHECK:   IERT.ExecutorResource 1 of "DMA_UPA"
// CHECK:   IERT.ExecutorResource 1 of "DMA_NN"
// CHECK: } usedExecutors :  {
// CHECK: }

}
