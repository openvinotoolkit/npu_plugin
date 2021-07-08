// RUN: vpux-opt --vpux-edsl-autotile="min-count=3 max-count=3 min-inner-buffer=1024 processing-tags=middle outer-tags=middle inner-tags=inner" %s | FileCheck %s

func @eltwise_sum(%arg0: memref<1x3x64x64xf16>, %arg1: memref<1x3x64x64xf16>) -> memref<1x3x64x64xf16> {
  %0 = memref.alloc() : memref<1x3x64x64xf16>
  %1 = affine.parallel (%arg2, %arg3, %arg4, %arg5) = (0, 0, 0, 0) to (1, 3, 64, 64) step (1, 3, 64, 64) reduce ("assign") -> (memref<1x3x64x64xf16>) {
    %2 = affine.parallel (%arg6, %arg7, %arg8, %arg9) = (%arg2, %arg3, %arg4, %arg5) to (%arg2 + 1, %arg3 + 3, %arg4 + 64, %arg5 + 64) reduce ("assign") -> (memref<1x3x64x64xf16>) {
      %3 = pxa.load %arg1[0, %arg7, %arg8, %arg9] : memref<1x3x64x64xf16>
      %4 = pxa.load %arg0[0, %arg7, %arg8, %arg9] : memref<1x3x64x64xf16>
      %5 = addf %3, %4 : f16 
      %6 = pxa.reduce assign %5, %0[%arg6, %arg7, %arg8, %arg9] : memref<1x3x64x64xf16>
      affine.yield %6 : memref<1x3x64x64xf16>
    } {tags = {middle}}
    affine.yield %2 : memref<1x3x64x64xf16>
  } {tags = {outer, outermost}}
  return %1 : memref<1x3x64x64xf16>
}

// CHECK-LABEL: @eltwise_sum
// CHECK: (%[[arg0:.*]]: memref<1x3x64x64xf16>, %[[arg1:.*]]: memref<1x3x64x64xf16>)
// CHECK:   %[[out:.*]] = memref.alloc()
// CHECK:   affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (1, 3, 64, 64)
// CHECK-SAME: step (1, 3, 64, 64)
// CHECK:     affine.parallel (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]]) =
// CHECK-SAME: (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]) to
// CHECK-SAME: (%[[arg2]] + 1, %[[arg3]] + 3, %[[arg4]] + 64, %[[arg5]] + 64) step (1, 1, 64, 64)
// CHECK:       affine.parallel (%[[arg10:.*]], %[[arg11:.*]], %[[arg12:.*]], %[[arg13:.*]]) =
// CHECK-SAME: (%[[arg6]], %[[arg7]], %[[arg8]], %[[arg9]]) to
// CHECK-SAME: (%[[arg6]] + 1, %[[arg7]] + 1, %[[arg8]] + 64, %[[arg9]] + 64)
// CHECK:         pxa.load %[[arg1]][0, %[[arg11]], %[[arg12]], %[[arg13]]]
// CHECK:         pxa.load %arg0[0, %[[arg11]], %[[arg12]], %[[arg13]]]
// CHECK:         pxa.reduce assign {{.*}}, %[[out]][%[[arg10]], %[[arg11]], %[[arg12]], %[[arg13]]]
// CHECK:       tags = {inner}
// CHECK:     tags = {middle}
// CHECK:   tags = {outer, outermost}
