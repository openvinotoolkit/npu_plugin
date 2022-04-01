// RUN: vpux-opt --split-input-file --convert-sw-layers-to-VPUIP %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]


// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @SingleSWLayer(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %0: memref<1x1x1x1000xf16>

// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = IERT.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs([[VAR0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs([[VAR1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>):
// CHECK:   VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK: return [[VAR4]] : memref<1x1x1x1000xf16>

}

}

// -----

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]


// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_MemPermute(memref<*xf16>, memref<*xf16>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @MemPermuteSWLayer(%arg0: memref<1x2x3x4xf16>, %arg1: memref<1x3x4x2xf16>) -> memref<1x3x4x2xf16> {
    %0 = IERT.MemPermute {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} inputs(%arg0 : memref<1x2x3x4xf16>) outputs(%arg1 : memref<1x3x4x2xf16>) -> memref<1x3x4x2xf16>
    return %0: memref<1x3x4x2xf16>

// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = IERT.Copy inputs(%arg0 : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x3x4x2xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_MemPermute inputs([[VAR1]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x3x4x2xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x2xf16, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg2: memref<1x2x3x4xf16, [@CMX_NN, 0]>, %arg3: memref<1x3x4x2xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg2, %arg3) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x3x4x2xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x3x4x2xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x4x2xf16>) -> memref<1x3x4x2xf16>
// CHECK: return [[VAR4]] : memref<1x3x4x2xf16>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]


// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_MemPermute(memref<*xf16>, memref<*xf16>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @ReorderSWLayer(%arg0: memref<1x2x3x4xf16>, %arg1: memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC> {
    %0 = IERT.MemPermute {mem_perm = #NHWC} inputs(%arg0 : memref<1x2x3x4xf16>) outputs(%arg1 : memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC>
    return %0: memref<1x2x3x4xf16, #NHWC>

// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = IERT.Copy inputs(%arg0 : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_MemPermute inputs([[VAR1]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg2: memref<1x2x3x4xf16, [@CMX_NN, 0]>, %arg3: memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>):  // no predecessors
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg2, %arg3) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC>
// CHECK: return [[VAR4]] : memref<1x2x3x4xf16, #NHWC>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {
// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
// CHECK: module @VPU.SW  {
// CHECK-NEXT: func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
// CHECK-NEXT: func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT: func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK: func @ThreeSWLayers(%arg0: memref<1x1x1x2000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
func @ThreeSWLayers(%arg0: memref<1x1x1x2000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %tmp1 = memref.alloc() : memref<1x1x1x2000xf16>
    %tmp2 = memref.alloc() : memref<1x1x1x2000xf16>

    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1x1x1x2000xf16>) outputs(%tmp1 : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>
    %1 = IERT.Sigmoid {axisInd = 3} inputs(%0 : memref<1x1x1x2000xf16>) outputs(%tmp2 : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

    %2 = IERT.SubView %1[0, 0, 0, 1000] [1, 1, 1, 1000]
        : memref<1x1x1x2000xf16> to memref<1x1x1x1000xf16, {order = #NCHW, strides = [2000, 2000, 2000, 1]}>

    %3 = IERT.SoftMax {axisInd = 3} inputs(%2 : memref<1x1x1x1000xf16, {order = #NCHW, strides = [2000, 2000, 2000, 1]}>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>

    return %3 : memref<1x1x1x1000xf16>

// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>
// CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = IERT.Copy inputs(%arg0 : memref<1x1x1x2000xf16>) outputs([[VAR2]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs([[VAR3]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>  {
// CHECK:       ^bb0(%arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>, %arg3: memref<1x1x1x2000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:         VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:       }
// CHECK: [[VAR6:%.*]] = IERT.Copy inputs([[VAR5]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR0]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR7:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR8:%.*]] = IERT.Copy inputs([[VAR6]] : memref<1x1x1x2000xf16>) outputs([[VAR7]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR9:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR10:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_Sigmoid inputs([[VAR8]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR9]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>  {
// CHECK:      ^bb0(%arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>, %arg3: memref<1x1x1x2000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:       VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR11:%.*]] = IERT.Copy inputs([[VAR10]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR1]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR12:%.*]] = IERT.SubView [[VAR11]] [0, 0, 0, 1000] [1, 1, 1, 1000] : memref<1x1x1x2000xf16> to memref<1x1x1x1000xf16, {order = #NCHW, strides = [2000, 2000, 2000, 1]}>

// CHECK: [[VAR13:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR14:%.*]] = IERT.Copy inputs([[VAR12]] : memref<1x1x1x1000xf16, {order = #NCHW, strides = [2000, 2000, 2000, 1]}>) outputs([[VAR13]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR15:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR16:%.*]] = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs([[VAR14]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>  {
// CHECK:  ^bb0(%arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:  VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR17:%.*]] = IERT.Copy inputs([[VAR16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>

// CHECK:  return [[VAR17]] : memref<1x1x1x1000xf16>

}

}
