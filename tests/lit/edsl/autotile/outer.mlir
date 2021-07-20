// RUN: vpux-opt --vpux-edsl-autotile="min-count=1 max-count=3 min-inner-buffer=3072 processing-tags=outermost outer-tags=outer inner-tags=middle" %s | FileCheck %s

func @eltwise_sum(%arg0: memref<1x3x64x64xf16>, %arg1: memref<1x3x64x64xf16>) -> memref<1x3x64x64xf16> {
  %0 = memref.alloc() : memref<1x3x64x64xf16>
  %1 = affine.parallel (%arg2, %arg3, %arg4, %arg5) = (0, 0, 0, 0) to (1, 3, 64, 64) reduce("assign") -> (memref<1x3x64x64xf16>) {
    %2 = pxa.load %arg1[0, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    %3 = pxa.load %arg0[0, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    %4 = addf %2, %3 : f16
    %5 = pxa.reduce assign %4, %0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    affine.yield %5 : memref<1x3x64x64xf16>
  } {tags = {outermost}}
  return %1 : memref<1x3x64x64xf16>
}

// CHECK-LABEL: @eltwise_sum
// CHECK: (%[[arg0:.*]]: memref<1x3x64x64xf16>, %[[arg1:.*]]: memref<1x3x64x64xf16>)
// CHECK:   %[[out:.*]] = memref.alloc()
// CHECK:   affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (1, 3, 64, 64)
// CHECK-SAME: step (1, 1, 64, 64)
// CHECK:     affine.parallel (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]]) =
// CHECK-SAME: (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]) to
// CHECK-SAME: (%[[arg2]] + 1, %[[arg3]] + 1, %[[arg4]] + 64, %[[arg5]] + 64)
// CHECK:       pxa.load %[[arg1]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK:       pxa.load %[[arg0]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK:       pxa.reduce assign {{.*}}, %[[out]][%[[arg6]], %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK:     tags = {middle}
// CHECK:   tags = {outer, outermost}
