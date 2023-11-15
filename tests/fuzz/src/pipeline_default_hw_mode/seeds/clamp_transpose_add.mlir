#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x30x30x30xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x30x30x30xf16>
  }

  func.func @main(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 20.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %1 = IE.Clamp(%0) {max = 10.000000e+00 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %3 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x30x30x30xf16>, tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %4 : tensor<1x30x30x30xf16>
  }
}
