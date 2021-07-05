// RUN: vpux-opt --vpux-edsl-autotile="min-count=3 max-count=65536 processing-tags=outermost outer-tags=outer inner-tags=inner no-negative-index=true" %s | FileCheck %s

#set0 = affine_set<(d0, d1, d2) : (d0 - d1 * 64 - d2 * 128 >= 0, -d0 + d1 * 64 + d2 * 128 + 63 >= 0)>

func @plaidml_kernel_0(%arg0: memref<1x64x26x26xf16>, %arg1: memref<1x256x13x13xf16>) {
  %0 = affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (256, 13, 13, 2, 2) reduce ("assign") -> (memref<1x256x13x13xf16>) {
    %1 = affine.if #set0(%arg2, %arg5, %arg6) -> memref<1x256x13x13xf16> {
      %2 = pxa.load %arg0[0, %arg2 - %arg5 * 64 - %arg6 * 128, %arg3 * 2 + %arg6, %arg4 * 2 + %arg5] : memref<1x64x26x26xf16>
      %3 = pxa.reduce assign %2, %arg1[0, %arg2, %arg3, %arg4] : memref<1x256x13x13xf16>
      affine.yield %3 : memref<1x256x13x13xf16>
    } else {
      affine.yield %arg1 : memref<1x256x13x13xf16>
    }
    affine.yield %1 : memref<1x256x13x13xf16>
  } {tags = {outermost}}
  return
}

// CHECK-LABEL: @plaidml_kernel_0
// CHECK: (%[[arg0:.*]]: memref<1x64x26x26xf16>, %[[arg1:.*]]: memref<1x256x13x13xf16>)
// CHECK:   affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]]) = (0, 0, 0, 0, 0) to
// CHECK-SAME: (256, 13, 13, 2, 2) step (32, 13, 13, 1, 1)
// CHECK:     affine.parallel (%[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]], %[[arg10:.*]], %[[arg11:.*]]) =
// CHECK-SAME: (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]], %[[arg6]]) to
// CHECK-SAME: (%[[arg2]] + 32, %[[arg3]] + 13, %[[arg4]] + 13, %[[arg5]] + 1, %[[arg6]] + 1)
// CHECK:       affine.if
// CHECK:         pxa.load %[[arg0]][0, %[[arg7]] - %[[arg10]] * 64 - %[[arg11]] * 128, %[[arg8]] * 2 + %[[arg11]],
// CHECK-SAME: %[[arg9]] * 2 + %[[arg10]]]
// CHECK:         pxa.reduce assign {{.*}}, %[[arg1]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK:         affine.yield
// CHECK:       else
// CHECK:         affine.yield
// CHECK:       affine.yield
// CHECK:     tags = {inner}
// CHECK:     affine.yield
// CHECK:   tags = {outer, outermost}
