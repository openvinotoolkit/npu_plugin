// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s

// CHECK-LABEL: @ConvertReduceMeanToPooling4D
func @ConvertReduceMeanToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>

  // CHECK:       %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPooling3D
func @ConvertReduceMeanToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [256, 1, 7, 7]} : tensor<256x7x7xf16> -> tensor<256x1x7x7xf16>
  // CHECK-NOT:   ReduceMean
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<256x1x7x7xf16> -> tensor<256x1x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<256x1x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOneKeepDim
func @ConvertReduceMeanToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOne
func @ConvertReduceMeanToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMean(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMaxToPooling4D
func @ConvertReduceMaxToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>
  // CHECK:       %0 = IE.MaxPool(%arg0) {kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPooling3D
func @ConvertReduceMaxToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [256, 1, 7, 7]} : tensor<256x7x7xf16> -> tensor<256x1x7x7xf16>
  // CHECK-NOT:   ReduceMax
  // CHECK:       %1 = IE.MaxPool(%0) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<256x1x7x7xf16> -> tensor<256x1x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<256x1x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOneKeepDim
func @ConvertReduceMaxToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOne
func @ConvertReduceMaxToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceMax(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceSumToPooling4D
func @ConvertReduceSumToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>

  // CHECK:       %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       %cst_0 = const.Declare tensor<1xf16> = #const.Content<dense<5.000000e+01> : tensor<1xf16>>
  // CHECK:       %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = "NUMPY"} : tensor<1x1x1x1xf16>, tensor<1xf16> -> tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPooling3D
func @ConvertReduceSumToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [256, 1, 7, 7]} : tensor<256x7x7xf16> -> tensor<256x1x7x7xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<256x1x7x7xf16> -> tensor<256x1x1x7xf16>
  // CHECK:       %cst_0 = const.Declare tensor<1xf16> = #const.Content<dense<7.000000e+00> : tensor<1xf16>>
  // CHECK:       %2 = IE.Multiply(%1, %cst_0) {auto_broadcast = "NUMPY"} : tensor<256x1x1x7xf16>, tensor<1xf16> -> tensor<256x1x1x7xf16>
  // CHECK:       %3 = IE.Reshape(%2) {shape_value = [256, 1, 7]} : tensor<256x1x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOneKeepDim
func @ConvertReduceSumToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOne
func @ConvertReduceSumToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %1 = IE.ReduceSum(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @NotConvertReduceMean
func @NotConvertReduceMean(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %cst = const.Declare tensor<1xsi32> = #const.Content<dense<[3]> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
  %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
  // CHECK-NOT:   AvgPool
}
