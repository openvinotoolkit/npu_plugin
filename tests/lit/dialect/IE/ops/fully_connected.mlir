// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @UseFullyConnected
func @UseFullyConnected(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = #const.Content<dense<1.0> : tensor<64x16xf32>>
    %0 = IE.MatMul(%arg0, %weights)
        { transpose_b } :
        tensor<1x16xf32>, tensor<64x16xf32> -> tensor<1x64xf32>
    return %0 : tensor<1x64xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf32> = #const.Content<dense<1.000000e+00> : tensor<64x16xf32>>
    // CHECK:       %[[VAL0:.*]] = IE.FullyConnected(%arg0, %[[WEIGHTS]]) : tensor<1x16xf32>, tensor<64x16xf32> -> tensor<1x64xf32>
    // CHECK-NOT:   IE.MatMul
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @FuseFCAndBias
func @FuseFCAndBias(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = #const.Content<dense<1.0> : tensor<64x16xf32>>
    %0 = IE.FullyConnected(%arg0, %weights) :
        tensor<1x16xf32>, tensor<64x16xf32> -> tensor<1x64xf32>

    %bias = const.Declare tensor<1x64xf32> = #const.Content<dense<1.0> : tensor<1x64xf32>>
    %1 = IE.Add(%0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x64xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %1 : tensor<1x64xf32>

    // CHECK-DAG:   %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf32> = #const.Content<dense<1.000000e+00> : tensor<64x16xf32>>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x64xf32> = #const.Content<dense<1.000000e+00> : tensor<1x64xf32>>
    // CHECK:       %[[VAL0:.*]] = IE.FullyConnected(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @UseFullyConnectedWithTransposedWeights
func @UseFullyConnectedWithTransposedWeights(%arg0: tensor<1x512xf32>) -> tensor<1x40xf32> {
    %cst_0 = const.Declare tensor<512x40xf32> = #const.Content<dense<1.0> : tensor<512x40xf32>>
    %1 = IE.MatMul(%arg0, %cst_0) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    return %1 : tensor<1x40xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<40x512xf32> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x40xf32>, [#const.Transpose<#map>]>
    // CHECK:       %[[FC_OUT:.*]] = IE.FullyConnected(%arg0, %[[WEIGHTS]]) : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>
    // CHECK:       return %[[FC_OUT]] : tensor<1x40xf32>
}
