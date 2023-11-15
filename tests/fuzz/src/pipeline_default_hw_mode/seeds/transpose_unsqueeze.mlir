#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x28x70xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x28x70xf16>
  }

  func.func @main(%arg0: tensor<1x28x70xf16>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>
    %1 = IE.Unsqueeze(%0) {axes_value = [0]} : tensor<1x70x28xf16> -> tensor<1x1x70x28xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x1x70x28xf16> -> tensor<1x1x28x70xf16>
    return %2 : tensor<1x1x28x70xf16>
  }
}
