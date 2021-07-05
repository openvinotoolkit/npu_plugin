// RUN: vpux-opt --vpux-edsl-autotile="min-count=65536 max-count=16777216 output-indices-only min-inner-buffer=0 processing-tags=outermost outer-tags=outer inner-tags=middle" %s | FileCheck %s

func @edsl_conv(%arg0: memref<5x5x5x8xf16>, %arg1: memref<1x256x256x3xf16>) -> memref<1x256x256x8xf16> {
  %0 = memref.alloc() : memref<1x256x256x8xf16>
  %1 = affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0, 0, 0) to (3, 8, 5, 5, 256, 256) reduce("assign") -> (memref<1x256x256x8xf16>) {
    %2 = pxa.load %arg1[0, %arg4 + %arg6 - 2, %arg5 + %arg7 - 2, %arg2] : memref<1x256x256x3xf16>
    %3 = pxa.load %arg0[%arg4, %arg5, %arg2, %arg3] : memref<5x5x5x8xf16>
    %4 = mulf %2, %3 : f16
    %5 = pxa.reduce addf %4, %0[0, %arg6, %arg7, %arg3] : memref<1x256x256x8xf16>
    affine.yield %5 : memref<1x256x256x8xf16>
  } {tags = {outermost}}
  return %1 : memref<1x256x256x8xf16>
}

// CHECK-LABEL: @edsl_conv
// CHECK: (%[[arg0:.*]]: memref<5x5x5x8xf16>, %[[arg1:.*]]: memref<1x256x256x3xf16>)
// CHECK:   %[[out:.*]] = memref.alloc() 
// CHECK:   affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) =
// CHECK-SAME: (0, 0, 0, 0, 0, 0) to (3, 8, 5, 5, 256, 256) step (3, 8, 5, 5, 1, 1)
// CHECK:     affine.parallel (%[[arg8:.*]], %[[arg9:.*]], %[[arg10:.*]], %[[arg11:.*]], %[[arg12:.*]], %[[arg13:.*]]) =
// CHECK-SAME: (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]], %[[arg6]], %[[arg7]]) to
// CHECK-SAME: (%[[arg2]] + 3, %[[arg3]] + 8, %[[arg4]] + 5, %[[arg5]] + 5, %[[arg6]] + 1, %[[arg7]] + 1)
// CHECK:       pxa.load %[[arg1]][0, %[[arg10]] + %[[arg12]] - 2, %[[arg11]] + %[[arg13]] - 2, %[[arg8]]]
// CHECK:       pxa.load %[[arg0]][%[[arg10]], %[[arg11]], %[[arg8]], %[[arg9]]]
// CHECK:       pxa.reduce addf {{.*}}, %[[out]][0, %[[arg12]], %[[arg13]], %[[arg9]]]
