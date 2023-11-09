#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  IE.CNNNetwork entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x4x8x64xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x4x8x64xf32, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32, {order = #NHWC}> {
    %0 = IE.LayoutCast(%arg0) {dst_order = #NCWH} : tensor<1x4x8x64xf32> -> tensor<1x4x8x64xf32, {order = #NCWH}>
    %1 = IE.LayoutCast(%0) {dst_order = #NHWC} : tensor<1x4x8x64xf32, {order = #NCWH}> -> tensor<1x4x8x64xf32, {order = #NHWC}>
    return %1 : tensor<1x4x8x64xf32, {order = #NHWC}>
  }
}
