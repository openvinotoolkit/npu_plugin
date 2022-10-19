// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s

// CHECK-LABEL: @ConvertReduceMean
func @ConvertReduceMean(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<[3]> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       [[VAL0:%.*]] = IE.AvgPool(%arg0)
  // CHECK-SAME:    exclude_pads
  // CHECK-SAME:    kernel_size = [1, 112]
  // CHECK-SAME:    pads_begin = [0, 0]
  // CHECK-SAME:    pads_end = [0, 0]
  // CHECK-SAME:    rounding_type = "FLOOR"
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:    : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  // CHECK:       return [[VAL0]]
}
