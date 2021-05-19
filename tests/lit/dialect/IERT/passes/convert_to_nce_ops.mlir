// RUN: vpux-opt --split-input-file --convert-to-nce-ops %s | FileCheck %s


module @MLIRConv2dTest attributes {VPUIP.arch = "VPU3700", VPUIP.compilationMode = "ReferenceHW"}  {
  VPUIP.Graph options : "NONE" version : {contextStr = "VPUX Compiler", hash = "", majorV = 3 : i32, minorV = 11 : i32, patchV = 0 : i32}
  IERT.RunTimeResources availableMemory :  {
    IERT.MemoryResource 1073741824 bytes
    IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
    IERT.MemoryResource 4194304 bytes of "CMX_UPA" {VPUIP.bandwidth = 16 : i64, VPUIP.derateFactor = 8.500000e-01 : f64}
    IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
  } executors :  {
    IERT.ExecutorResource 1 of "Leon_RT"
    IERT.ExecutorResource 1 of "Leon_NN"
    IERT.ExecutorResource 1 of "DMA_UPA"
    IERT.ExecutorResource 16 of "SHAVE_UPA"
    IERT.ExecutorResource 20 of "SHAVE_NN"
    IERT.ExecutorResource 4 of "NCE_Cluster"  {
      IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
    }
    IERT.ExecutorResource 1 of "DMA_NN"
  }
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    IE.DataInfo "Parameter_0" : memref<1x16x16x16xf16>
  } outputsInfo :  {
    IE.DataInfo "Add_4" : memref<1x16x16x16xf16>
  }
  func @main(%arg0: memref<1x16x16x16xf16>, %arg1: memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16> {
    %0 = IERT.Constant memref<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %1 = IERT.Constant memref<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %2 = memref.alloc() : memref<1x16x16x16xf16>
    %3 = IERT.Convolution {dilations = [1 : i32, 1 : i32], pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} inputs(%arg0 : memref<1x16x16x16xf16>, %0 : memref<16x16x1x1xf16>, %1 : memref<1x16x1x1xf16>) outputs(%2 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>
    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16>) outputs(%arg1 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>
    return %4 : memref<1x16x16x16xf16>
  }

// CHECK:    %0 = IERT.Constant memref<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
// CHECK:    %1 = memref.alloc() : memref<1x16x16x16xf16>
// CHECK:    %2 = memref.alloc() : memref<16x16x1x1xf16, #NHWC>
// CHECK:    %3 = IERT.Reorder inputs(%0 : memref<16x16x1x1xf16>) outputs(%2 : memref<16x16x1x1xf16, #NHWC>) -> memref<16x16x1x1xf16, #NHWC>
// CHECK:    %4 = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:    %5 = IERT.Copy inputs(%3 : memref<16x16x1x1xf16, #NHWC>) outputs(%4 : memref<16x16x1x1xf16, #NHWC, "CMX_NN">) -> memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:    %6 = IERT.Constant memref<16x1x1x4xsi32, #NHWC> = dense{{.*}} : tensor<16x1x1x4xsi32>
// CHECK:    %7 = memref.alloc() : memref<16x1x1x4xsi32, #NHWC, "CMX_NN">
// CHECK:    %8 = IERT.Copy inputs(%6 : memref<16x1x1x4xsi32, #NHWC>) outputs(%7 : memref<16x1x1x4xsi32, #NHWC, "CMX_NN">) -> memref<16x1x1x4xsi32, #NHWC, "CMX_NN">
// CHECK:    %9 = memref.alloc() : memref<1x16x16x16xf16, #NHWC>
// CHECK:    %10 = IERT.Reorder inputs(%arg0 : memref<1x16x16x16xf16>) outputs(%9 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>
// CHECK:    %11 = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:    %12 = IERT.Copy inputs(%10 : memref<1x16x16x16xf16, #NHWC>) outputs(%11 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) -> memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:    %13 = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:    %14 = VPUIP.NCEClusterTask {fixed_ppe_task = "NOOP", kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], kernel_size = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32], task_type = "CONV"} inputs(%12 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">, %5 : memref<16x16x1x1xf16, #NHWC, "CMX_NN">, %8 : memref<16x1x1x4xsi32, #NHWC, "CMX_NN">)) parent_input(%12 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) parent_output(%13 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) outputs(%13 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) variants :  {
// CHECK:    VPUIP.DPUTask {end = [15 : i32, 15 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 0 : i32, 0 : i32]}
// CHECK:    } -> memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:    %15 = memref.alloc() : memref<1x16x16x16xf16, #NHWC>
// CHECK:    %16 = IERT.Copy inputs(%14 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) outputs(%15 : memref<1x16x16x16xf16, #NHWC>) -> memref<1x16x16x16xf16, #NHWC>
// CHECK:    %17 = IERT.Reorder inputs(%16 : memref<1x16x16x16xf16, #NHWC>) outputs(%1 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>
// CHECK:    %18 = IERT.Copy inputs(%17 : memref<1x16x16x16xf16>) outputs(%arg1 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>
// CHECK:    return %18 : memref<1x16x16x16xf16>

}

