module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x3x300x300xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x300x300xf32>
  }

  func.func @main(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %0)
      {
          strides = [1, 1],
          pads_begin = [1, 1],
          pads_end = [1, 1],
          dilations = [1, 1]
      } :
      tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %2 = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %3 = IE.ScaleShift(%1, %2)
      {operandSegmentSizes = array<i32: 1, 0, 1>} :
      tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %3 : tensor<1x16x300x300xf32>
  }
}
