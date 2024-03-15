module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x3x10x10xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x20x15xf32>
  }

  func.func @main(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
  }
}
