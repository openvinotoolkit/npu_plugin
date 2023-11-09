module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x5x10x11xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x11x12x12xf16>
  }

  func.func @main(%arg0: tensor<1x5x10x11xf16>) -> tensor<1x11x12x12xf16> {
    %0 = const.Declare tensor<4xsi64> = dense<[0, 3, 0, 1]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xsi64> = dense<[0, 3, 2, 0]> : tensor<4xsi64>
    %2 = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    %3 = IE.Pad(%arg0)[%0, %1, %2] {mode = #IE.pad_mode<SYMMETRIC>} : tensor<1x5x10x11xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<f16> -> tensor<1x11x12x12xf16>
    return %3 : tensor<1x11x12x12xf16>
  }
}
