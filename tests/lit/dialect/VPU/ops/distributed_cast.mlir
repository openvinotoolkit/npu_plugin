// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedTensor = type !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Fold
func @Fold(%arg0: tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to !DistributedTensor
    %1 = VPU.DistributedCast(%0 : !DistributedTensor) -> !DistributedTensor
    %2 = builtin.unrealized_conversion_cast %1 : !DistributedTensor to tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %2 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-NOT:  VPU.DistributedCast
    // CHECK:      return %arg0 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
}
