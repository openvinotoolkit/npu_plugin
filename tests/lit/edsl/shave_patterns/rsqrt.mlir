// RUN: vpux-opt --vpux-edsl-shave-patterns %s | FileCheck %s

func @rsqrt(%arg0: memref<8x8xf16>, %arg1: memref<8x8xf16>, %arg2: memref<8x8xf16>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %0 = memref.load %arg0[%c0, %c1] : memref<8x8xf16>
  %1 = math.sqrt %0 : f16
  %2 = math.sqrt %1 : f16
  scf.for %arg3 = %c0 to %c8 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %3 = memref.load %arg0[%arg3, %arg4] : memref<8x8xf16>
      %4 = divf %3, %2 : f16
      memref.store %4, %arg0[%arg3, %arg4] : memref<8x8xf16>
      %5 = memref.load %arg1[%arg3, %arg4] : memref<8x8xf16>
      %6 = divf %5, %2 : f16
      memref.store %6, %arg1[%arg3, %arg4] : memref<8x8xf16>
      %7 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf16>
      %8 = divf %2, %7 : f16
      memref.store %8, %arg2[%arg3, %arg4] : memref<8x8xf16>
      scf.yield
    }
    scf.yield
  }
  %9 = memref.load %arg0[%c0, %c1] : memref<8x8xf16>
  %10 = math.sqrt %9 : f16
  %11 = divf %9, %10 : f16
  memref.store %11, %arg0[%c0, %c1] : memref<8x8xf16>
  return
}

// CHECK-LABEL: func @rsqrt
// CHECK:         memref.load
// CHECK:         math.sqrt
// CHECK:         math.sqrt
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK-NEXT:        memref.load
// CHECK-NEXT:        math.rsqrt
// CHECK-NEXT:        mulf
// CHECK-NEXT:        memref.store
// CHECK-NEXT:        memref.load
// CHECK-NEXT:        math.rsqrt
// CHECK-NEXT:        mulf
// CHECK-NEXT:        memref.store
// CHECK-NEXT:        memref.load
// CHECK-NEXT:        divf
// CHECK-NEXT:        memref.store
// CHECK:         memref.load
// CHECK-NEXT:    math.rsqrt
// CHECK-NEXT:    mulf
