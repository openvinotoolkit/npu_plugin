// RUN: vpux-opt --vpux-edsl-autotile="min-count=3 max-count=65536 min-inner-buffer=3072 processing-tags=middle outer-tags=middle inner-tags=inner" %s | FileCheck %s

// CHECK-LABEL: @grn
func @grn(%arg0: memref<1x224x128x24xf16>) -> memref<1x224x128x24xf16> {
// CHECK: (%[[arg0:.*]]: memref<1x224x128x24xf16>)
  %cst = constant 1.001360e-05 : f16
  %cst_0 = constant 0.000000e+00 : f16
  %cst_1 = constant 1.000000e+00 : f16
  %0 = memref.alloc() : memref<1x224x128x24xf16>
  // CHECK: %[[out:.*]] = memref.alloc()
  %1 = affine.parallel (%arg1, %arg2) = (0, 0) to (224, 128) step (224, 128) reduce ("assign") -> (memref<1x224x128x24xf16>) {
    // CHECK: affine.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (0, 0) to (224, 128) step (224, 128)
    %2 = affine.parallel (%arg3, %arg4) = (%arg1, %arg2) to (%arg1 + 224, %arg2 + 128) reduce ("assign") -> (memref<1x224x128x24xf16>) {
    // CHECK: affine.parallel (%[[arg3:.*]], %[[arg4:.*]]) = (%[[arg1]], %[[arg2]]) to (%[[arg1]] + 224, %[[arg2]] + 128) step (4, 128)
    // CHECK: affine.parallel (%[[arg5:.*]], %[[arg6:.*]]) = (%[[arg3]], %[[arg4]]) to (%[[arg3]] + 4, %[[arg4]] + 128)
      %3 = memref.alloc() : memref<1x1x1x1xf16>
      %4 = pxa.reduce assign %cst_0, %3[0, 0, 0, 0] : memref<1x1x1x1xf16>
      %5 = affine.parallel (%arg5) = (0) to (24) reduce ("assign") -> (memref<1x1x1x1xf16>) {
      // CHECK: affine.parallel (%[[arg7:.*]]) = (0) to (24)
        %12 = pxa.load %arg0[0, %arg3, %arg4, %arg5] : memref<1x224x128x24xf16>
        // CHECK: pxa.load %arg0[0, %[[arg5]], %[[arg6]], %[[arg7]]]
        %13 = pxa.load %arg0[0, %arg3, %arg4, %arg5] : memref<1x224x128x24xf16>
        // CHECK: pxa.load %arg0[0, %[[arg5]], %[[arg6]], %[[arg7]]]
        %14 = mulf %12, %13 : f16
        %15 = pxa.reduce addf %14, %4[0, 0, 0, 0] : memref<1x1x1x1xf16>
        affine.yield %15 : memref<1x1x1x1xf16>
      }
      %6 = pxa.load %5[0, 0, 0, 0] : memref<1x1x1x1xf16>
      %7 = addf %6, %cst_1 : f16
      %8 = math.sqrt %7 : f16
      %9 = cmpf "olt", %8, %cst : f16
      %10 = select %9, %cst, %8 : f16
      %11 = affine.parallel (%arg5) = (0) to (24) reduce ("assign") -> (memref<1x224x128x24xf16>) {
      // CHECK: affine.parallel (%[[arg7:.*]]) = (0) to (24)
        %12 = pxa.load %arg0[0, %arg3, %arg4, %arg5] : memref<1x224x128x24xf16>
        // CHECK: pxa.load %arg0[0, %[[arg5]], %[[arg6]], %[[arg7]]]
        %13 = divf %12, %10 : f16
        %14 = pxa.reduce assign %13, %0[0, %arg3, %arg4, %arg5] : memref<1x224x128x24xf16>
        // CHECK: pxa.reduce assign {{.*}}, %[[out]][0, %[[arg5]], %[[arg6]], %[[arg7]]]
        affine.yield %14 : memref<1x224x128x24xf16>
      }
      affine.yield %11 : memref<1x224x128x24xf16>
    } {tags = {middle}}
    affine.yield %2 : memref<1x224x128x24xf16>
  } {tags = {outer, outermost}}
  return %1 : memref<1x224x128x24xf16>
}
