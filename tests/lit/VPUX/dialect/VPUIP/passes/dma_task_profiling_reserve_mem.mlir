// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --dma-task-profiling-reserve-mem %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

module @SimpleGraph {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }

    // CHECK:         DmaProfilingReservedMemory
    // CHECK-NEXT:         IE.MemoryResource 256 bytes of @CMX_NN offset 0
}
