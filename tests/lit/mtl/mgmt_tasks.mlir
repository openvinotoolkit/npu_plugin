// This file generates a blob with management tasks, used to demonstrate that
// the runtime can process them.  It's also a lit test to help check for
// regressions in the VPUIP dialect.
//
// To generate a blob, use:
//
//    vpux-translate --export-VPUIP < mgmt_tasks.mlir > mgmt_tasks.blob
//
// RUN: vpux-opt %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f32, 1.000000e+00>

module @mainModule attributes {VPU.arch = "MTL", VPU.compilationMode = "ReferenceSW"} {

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 31457280 bytes of "DDR" {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
        IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}
    }
    usedMemory : {
    }
    executors : {
        IERT.ExecutorResource 1 of "DMA_NN"
        IERT.ExecutorResource 1 of  "SHAVE_NN"
        IERT.ExecutorResource 1 of  "SHAVE_ACT"
        IERT.ExecutorResource 1 of  "NCE" {
          IERT.ExecutorResource 1 of "DPU"
        }
    }

IE.CNNNetwork
    entryPoint : @mgmt_task_test
    inputsInfo : {
        DataInfo "input_0" : tensor<1x16x16x16xui8, {order = #NHWC}>
    }
    outputsInfo : {
        DataInfo "output_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
    }

func @mgmt_task_test(%arg0: memref<1x16x16x16x!qElemType, #NHWC>, %arg1: memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC> {
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%1 : !VPURT.Barrier) op : {
      VPUIP.Empty
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) op : {
      VPUIP.Empty
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) op : {
      VPUIP.Empty
    }

    return %arg1 : memref<1x16x16x16xf16, #NHWC>
  }
}

// CHECK-LABEL: module @mainModule attributes {VPU.arch = "MTL", VPU.compilationMode = "ReferenceSW"}
