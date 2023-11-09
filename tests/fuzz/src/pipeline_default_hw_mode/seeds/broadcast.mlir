module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x8x4x4xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x4x4xf32>
  }

  func.func @main(%arg0 : tensor<1x8x4x4xf32>)-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<1> : tensor<4xsi64>
    %1 = IE.Broadcast(%arg0, %0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x4x4xf32>, tensor<4xsi64> -> tensor<1x8x4x4xf32>
    return %1 : tensor<1x8x4x4xf32>
  }
}
