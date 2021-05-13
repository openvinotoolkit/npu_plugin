// RUN: vpux-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

// This test is here to assure that plaidml builds and that generic dialects
// and passes are registered with the vpux-opt tool.

func @eltwise_add(
  %arg0: tensor<10x20xf32>,
  %arg1: tensor<10x20xf32>
) -> tensor<10x20xf32> {
  %0 = tile.add %arg1, %arg0 : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @eltwise_add
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: addf
// CHECK: pxa.reduce assign
