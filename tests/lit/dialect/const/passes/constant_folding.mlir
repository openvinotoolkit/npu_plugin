// RUN: vpux-opt --split-input-file --constant-folding %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 0.0039215686274509803>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func @ConstFold() -> memref<16x3x1x1xf16, #map> {
    %0 = const.Declare memref<16x3x1x1xf16, #map> =
        #const.Content<dense<-1.0> : tensor<16x3x1x1xf32>,
        [
            #const.ConvertElemType<f16>,
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Dequantize,
            #const.Reorder<#map>
        ]>

    return %0 : memref<16x3x1x1xf16, #map>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1xf16, #map>
    // CHECK-SAME:       #const.Content<dense<
    // CHECK-SAME:       tensor<16x3x1x1xf16
    // CHECK-SAME:       {order = #map}>
    // CHECK:       return [[CST]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0039215686274509803>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func @QuantConstFold() -> memref<16x3x1x1x!qElemType, #map> {
    %0 = const.Declare memref<16x3x1x1x!qElemType, #map> =
        #const.Content<dense<129> : tensor<16x3x1x1xui8>,
        [
            #const.QuantCast<!qElemType>,
            #const.Reorder<#map>
        ]>

    return %0 : memref<16x3x1x1x!qElemType, #map>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1x!quant.uniform<u8:f16, 0.0039215686274509803>, #map>
    // CHECK-SAME:       #const.Content<dense<
    // CHECK-SAME:       tensor<16x3x1x1xui8
    // CHECK-SAME:       {order = #map}>
    // CHECK:       return [[CST]]
}
