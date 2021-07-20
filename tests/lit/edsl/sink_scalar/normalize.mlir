// RUN: vpux-opt --vpux-edsl-sink-scalar %s | FileCheck %s

// CHECK-LABEL: @edsl_normalize
func @edsl_normalize(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
  %cst = constant 0.000000e+00 : f16
  %cst_0 = constant 1.001360e-05 : f16
  %0 = memref.alloc() : memref<f16>
  %1 = pxa.reduce assign %cst, %0[] : memref<f16>
  %2 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (256, 256, 16) reduce ("assign") -> (memref<f16>) {
    %7 = pxa.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    %8 = mulf %7, %7 : f16
    %9 = pxa.reduce addf %8, %1[] : memref<f16>
    affine.yield %9 : memref<f16>
  }
  %3 = pxa.load %2[] : memref<f16>
  %4 = addf %3, %cst_0 : f16
  %5 = math.sqrt %4 : f16
  %6 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (256, 256, 16) reduce ("assign") -> (memref<1x256x256x16xf16>) {
    %7 = pxa.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    %8 = divf %7, %5 : f16
    %9 = pxa.reduce assign %8, %arg1[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    affine.yield %9 : memref<1x256x256x16xf16>
  }
  return %6 : memref<1x256x256x16xf16>
}

// CHECK:      memref.alloc
// CHECK:      pxa.reduce
// CHECK-NEXT: %[[r2:.*]] = affine.parallel
// CHECK:        pxa.load
// CHECK:        mulf
// CHECK:        pxa.reduce
// CHECK:        affine.yield
// CHECK:      affine.parallel
// CHECK-DAG:    constant
// CHECK-DAG:    pxa.load %[[r2]][]
// CHECK-DAG:    addf
// CHECK:        %[[r6:.*]] = math.sqrt
// CHECK:        %[[r7:.*]] = pxa.load
// CHECK:        divf %[[r7]], %[[r6]]
// CHECK:        pxa.reduce
// CHECK:        affine.yield
