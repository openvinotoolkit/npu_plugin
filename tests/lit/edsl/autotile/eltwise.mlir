// RUN: vpux-opt --vpux-edsl-autotile="min-count=1 total-buffer=120000" %s | FileCheck %s --check-prefix=CHECK-1
// RUN: vpux-opt --vpux-edsl-autotile="max-count=3" %s | FileCheck %s --check-prefix=CHECK-3

func @eltwise_sum(%arg0: memref<1x3x64x64xf16>, %arg1: memref<1x3x64x64xf16>) -> memref<1x3x64x64xf16> {
  %out = memref.alloc() : memref<1x3x64x64xf16>
  %0 = affine.parallel (%arg2, %arg3, %arg4, %arg5) = (0, 0, 0, 0) to (1, 3, 64, 64) reduce ("assign") -> (memref<1x3x64x64xf16>) {
    %1 = pxa.load %arg1[0, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    %2 = pxa.load %arg0[0, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    %3 = addf %1, %2 : f16
    %4 = pxa.reduce assign %3, %out[%arg2, %arg3, %arg4, %arg5] : memref<1x3x64x64xf16>
    affine.yield %4 : memref<1x3x64x64xf16>
  }
  return %0 : memref<1x3x64x64xf16>
}

// CHECK-1-LABEL: @eltwise_sum
// CHECK-1: (%[[arg0:.*]]: memref<1x3x64x64xf16>, %[[arg1:.*]]: memref<1x3x64x64xf16>)
// CHECK-1:   %[[out:.*]] = memref.alloc() : memref<1x3x64x64xf16>
// CHECK-1:   (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (1, 3, 64, 64) step (1, 3, 64, 64)
// CHECK-1:     (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]) to
// CHECK-1-SAME: (%[[arg2]] + 1, %[[arg3]] + 3, %[[arg4]] + 64, %[[arg5]] + 64)
// CHECK-1:       %[[arg1]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK-1:       %[[arg0]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK-1:       %[[out]][%[[arg6]], %[[arg7]], %[[arg8]], %[[arg9]]]


// CHECK-3-LABEL: @eltwise_sum
// CHECK-3: (%[[arg0:.*]]: memref<1x3x64x64xf16>, %[[arg1:.*]]: memref<1x3x64x64xf16>)
// CHECK-3:   %[[out:.*]] = memref.alloc() : memref<1x3x64x64xf16>
// CHECK-3:   (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (0, 0, 0, 0) to (1, 3, 64, 64) step (1, 1, 64, 64)
// CHECK-3:     (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]]) to
// CHECK-3-SAME: (%[[arg2]] + 1, %[[arg3]] + 1, %[[arg4]] + 64, %[[arg5]] + 64)
// CHECK-3:       %[[arg1]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK-3:       %[[arg0]][0, %[[arg7]], %[[arg8]], %[[arg9]]]
// CHECK-3:       %[[out]][%[[arg6]], %[[arg7]], %[[arg8]], %[[arg9]]]
