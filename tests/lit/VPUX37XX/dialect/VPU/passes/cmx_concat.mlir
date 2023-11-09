//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --cmx-concat --canonicalize %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsType = tensor<
    128x256x3x3xf16,
    {mem_space = @CMX_NN, order = #NHWC}
>

!WeightsTableType = tensor<
    128x1x1x4xsi32,
    {mem_space = @CMX_NN, order = #NCHW}
>

!SparseInput = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseOutput = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConvOutput = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {order = #NHWC}>
>

!SparseConvOutputCMX = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConvOutputDist = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConcatOutput = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {order = #NHWC}>
>

!SparseConcatOutputCMX = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

// CHECK:      func.func @SparseConvolution([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
// CHECK-SAME:                         [[WEIGHTS_TABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:                         [[WEIGHTS:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:         -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
func.func @SparseConvolution(%input: !SparseInput,
                        %weightsTable: !WeightsTableType,
                        %weights: !WeightsType)
        -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {

    // Convolution 0
    %0 = VPU.NCE.ClusterTiling (
        %input as %arg1: !SparseInput,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType) -> !SparseConvOutputDist {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                    -> !SparseConvOutputCMX
        VPU.Yield %20
    }

    // Convolution 0 output copy
    %1 = VPU.NCE.ClusterTiling (%0 as %arg1: !SparseConvOutputCMX) -> !SparseConvOutput {
            %20 = VPU.Copy(%arg1) : !SparseConvOutputCMX -> !SparseConvOutput
        VPU.Yield %20
    }

    // Convolution 1
    %2 = VPU.NCE.ClusterTiling (
        %input as %arg1: !SparseInput,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType)  -> !SparseConvOutputDist {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                    -> !SparseConvOutputCMX
        VPU.Yield %20
    }

    // Convolution 1 output copy
    %3 = VPU.NCE.ClusterTiling (%2 as %arg1: !SparseConvOutputCMX) -> !SparseConvOutput {
            %20 = VPU.Copy(%arg1) : !SparseConvOutputCMX -> !SparseConvOutput
        VPU.Yield %20
    }

    // Concat
    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : !SparseConvOutput, !SparseConvOutput -> !SparseConcatOutput

    // Concat output copy
    %5 = VPU.NCE.ClusterTiling (%4 as %arg1: !SparseConcatOutput) -> !SparseOutput {
        %20 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : !SparseConcatOutput -> !SparseConcatOutputCMX
        VPU.Yield %20
    }

    // Convolution 2
    %6 = VPU.NCE.ClusterTiling (
        %5 as %arg1: !SparseConcatOutputCMX,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType)
            -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]} -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>
        VPU.Yield %20
    }
    %7 = VPU.NCE.ClusterTiling (
        %6 as %arg1: !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>)
            -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>> {
        %20 = VPU.Copy(%arg1) : !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>> -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>
        VPU.Yield %20
    }

    return %7 : !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>

    // Convolution 0
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // Convolution 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // Concat
    // CHECK:       [[CONCAT_CMX:%.+]] = VPU.Concat([[CONV0]], [[CONV1]])

    // DistributedCast
    // CHECK:       [[DISTR_CAST:%.+]] = VPU.DistributedCast([[CONCAT_CMX]] : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:                    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Convolution 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.ClusterTiling ([[DISTR_CAST]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // CHECK:       [[COPY_OUT:%.+]] = VPU.NCE.ClusterTiling ([[CONV2]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
    // CHECK:           VPU.Copy([[INNER_ARG0]])

    // CHECK:       return [[COPY_OUT]]
}
