module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x16xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x64xf32>
  }

  func.func @main(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %0 = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %1 = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %2 = IE.FullyConnected(%arg0, %0, %1) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %2 : tensor<1x64xf32>
  }
}
