module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x8x4x4xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x4x4xf32>
  }

  func.func @main(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = -1} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %0 : tensor<1x8x4x4xf32>
  }
}
