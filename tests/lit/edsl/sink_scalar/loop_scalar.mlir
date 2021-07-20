// RUN: vpux-opt --vpux-edsl-sink-scalar %s | FileCheck %s

func @loop_scalar(%arg0: memref<4x4x4x4xf16>, %arg1: memref<4x4x4x4xf16>, %arg2: memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16> {
  %0 = memref.alloc() : memref<f16>
  %1 = affine.parallel (%arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0) to (4, 4, 4, 4) reduce ("assign") -> (memref<f16>) {
    %2 = pxa.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<4x4x4x4xf16>
    %3 = mulf %2, %2 : f16
    %4 = pxa.reduce addf %3, %0[] : memref<f16>
    affine.yield %4 : memref<f16>
  }
  %5 = pxa.load %1[] : memref<f16>
  %6 = pxa.load %arg0[0, 0, 0, 0] : memref<4x4x4x4xf16>
  %7 = addf %5, %6 : f16
  %8 = pxa.reduce assign %7, %arg2[0, 0, 0, 0] : memref<1x1x1x1xf16>
  return %8 : memref<1x1x1x1xf16>
}

// CHECK-LABEL: @loop_scalar
// CHECK: alloc
// CHECK: affine.parallel
// CHECK:   pxa.load
// CHECK:   mulf
// CHECK:   pxa.reduce
// CHECK:   affine.yield
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: addf
// CHECK-NEXT: %[[ret:.*]] = affine.parallel
// CHECK:   pxa.load
// CHECK:   pxa.load
// CHECK:   addf
// CHECK:   pxa.reduce
// CHECK:   affine.yield
// CHECK: return %[[ret]]
