// RUN: vpux-opt --init-compiler="vpu-arch=VPUX30XX" --dma-task-profiling-after-barrier %s | FileCheck %s

!dataType = type memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

module @DMAGraph {

  module @DmaProfilingReservedMemory {
    IE.MemoryResource 256 bytes of @CMX_NN offset 0
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo :  {
  }
  func @main(%arg0: !dataType, %arg1: !dataType) -> !dataType {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %buf0 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> !dataType
    %buf1 = VPURT.DeclareBuffer "CMX_NN" [0] <768> -> !dataType

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 10 : i64, isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataType) outputs(%buf0 : !dataType) -> !dataType
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 11 : i64, cycleEnd = 20 : i64, isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !dataType) outputs(%buf1 : !dataType) -> !dataType
    }

    VPURT.Task attributes {cycleBegin = 21 : i64, cycleEnd = 30 : i64, isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf1 : !dataType) outputs(%arg1 : !dataType) -> !dataType
    }

    return %arg1 : !dataType
  }
}

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:   DataInfo "dma" : tensor<6xui32>
// CHECK:        func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, 
// CHECK-SAME:       %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, 
// CHECK-SAME:       %arg2: memref<6xui32>) ->
// CHECK-SAME:       (memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       memref<6xui32>) {
// CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[BUF_DATA_0:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:    [[BUF_DATA_1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <768> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

// Getting start time of a DMA task
// CHECK:    [[REG_0:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_0:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_0]] :
// CHECK-SAME:        outputs([[PROF_BUF_0]] :

// Profiled DMA task
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs(%arg0 :
// CHECK-SAME:        outputs([[BUF_DATA_0]] :

// Getting end time of a DMA task
// CHECK:    [[REG_1:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <4> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task updates([[BAR0]]
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_1]]
// CHECK-SAME:        outputs([[PROF_BUF_1]]

// Getting start time of a DMA task
// CHECK:    [[REG_2:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_2:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task waits([[BAR0]]
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_2]] :
// CHECK-SAME:        outputs([[PROF_BUF_2]] :

// Profiled DMA task
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[BUF_DATA_0]] :
// CHECK-SAME:        outputs([[BUF_DATA_1]] :

// Getting end time of a DMA task
// CHECK:    [[REG_3:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_3:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <12> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_3]]
// CHECK-SAME:        outputs([[PROF_BUF_3]]

// Getting start time of a DMA task
// CHECK:    [[REG_4:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_4:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_4]] :
// CHECK-SAME:        outputs([[PROF_BUF_4]] :

// Profiled DMA task
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[BUF_DATA_1]] :
// CHECK-SAME:        outputs(%arg1 :

// Getting end time of a DMA task
// CHECK:    [[REG_5:%.+]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
// CHECK:    [[PROF_BUF_5:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <20> -> memref<1xui32, [@CMX_NN, 0]>
// CHECK:    VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[REG_5]]
// CHECK-SAME:        outputs([[PROF_BUF_5]]

// Copying prof buffer to DDR since no more tasks to profile
// CHECK:    [[PROF_BUF_CMX:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<6xui32, [@CMX_NN, 0]>
// CHECK:    [[PROF_OUTPUT:%.+]] = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<6xui32>
// CHECK:    VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64}
// CHECK-SAME:        inputs([[PROF_BUF_CMX]]
// CHECK-SAME:        outputs([[PROF_OUTPUT]]
