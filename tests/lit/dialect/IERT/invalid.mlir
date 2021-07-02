// RUN: vpux-opt %s --split-input-file --verify-diagnostics

module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo :  {
        IE.DataInfo "output" : memref<1x2x2x2xf16>
    }

func @main(%arg0: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
    %0 = const.Declare memref<1x2x2x2xf16> = #const.Content<dense<1.0> : tensor<1x2x2x2xf16>>
    %1 = IERT.Copy inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
// expected-error@+1 {{function output at index=0 should be an alias of the output buffer, but it's not}}
    return %0: memref<1x2x2x2xf16>
}

}
