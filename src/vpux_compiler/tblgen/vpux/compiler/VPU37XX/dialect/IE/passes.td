//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// InsertIdentityPoolBeforeOp
//

def InsertIdentityPoolBeforeOp : PassBase<"insert-identity-pool-before-op", "vpux::FunctionPass"> {
    let summary = "Insert Maxpool/AvgPool op before Activation ops and MemPermute ops";

    let description = [{
        The pass inserts MaxPool/AvgPool op before Activation ops and MemPermute ops.
        MaxPool/AvgPool will be lowered to a NCE op and the activation op will be fused into it.
        Supported activations: LeakyRelu, Clamp.
    }];

    let constructor = "vpux::IE::arch37xx::createInsertIdentityPoolBeforeOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MapBilinearInterpolateOnDPUPass
//

def MapBilinearInterpolateOnDPUPass : PassBase<"map-bilinear-interpolate-on-dpu", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, AvgPool and some depthwise convolution Ops";

    let description = [{
        Purpose:
        This pass replaces `Bilinear Interpolate` for which tiling is required to fit in CMX
        with sequences of operation that can be mapped on DPU and DMA.

        How it works:
        The supported interpolation axis currently supported are H and W.
        For each of these axis the scaling is happening individually, first perform vertical scaling and after perform horizontal scaling.
        On each axis the processing is split in three main regions BEGIN, MIDDLE and END.
        These three regions refers to slices from the output tensor and are influenced by the coordinate transformation mode attribute.
        * BEGIN - refers to the slice from output for which it is only needed to duplicate the first line/column from input
        * MIDDLE - refers to the slice from output where:
                    * for each output line/column from the output it is required to take two consecutive lines/colums from the input
                    * based on the coordinate transformation mode attribute compute the weight coefficients each of two lines/columns
                      has on theresulting output line/column
                    * each output line/column is computed with a GroupConvolution operation for which the weights are obtained by expanding
                     the weight coefficients of the input lines/columns
        * END - refers to the slice from output for which it is only needed to duplicate the last line/column from input
        ```
        Vertical scaling:                    Horizontal scaling
        ________________________       ____________________________
        |        BEGIN         |       |   |                  |   |
        |______________________|       |   |                  |   |
        |                      |       | B |        M         |   |
        |                      |       | E |        I         | E |
        |        MIDDLE        |       | G |        D         | N |
        |                      |       | I |        D         | D |
        |______________________|       | N |        L         |   |
        |         END          |       |   |        E         |   |
        |______________________|       |___|__________________|___|
        ```

        The rewrite implemented per each region is described below:
         BEGIN region:
        ```         Input
                      |
                    Slice
               first line/column
                |    ...    |
            Identity       Identity
            AvgPool        AvgPool

         MIDDLE region
                         Input
                  ---------|---------
                 |                   |
             Slice        ...       Slice
         two lines/colums       two lines/colums
               |                        |
           GroupConv               GroupConv
         one output line/colum   one output line/colum

         END region:
                    Input
                      |
                    Slice
               last line/column
                |    ...     |
            Identity       Identity
            AvgPool        AvgPool
        ```
        At the end the results of all the operation resulted are concatenated together on the scaling axis.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats.
    }];

    let constructor = "vpux::IE::arch37xx::createMapBilinearInterpolateOnDPUPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];


    let options = [
        Option<
            "interpolateAsSEOp", "interpolate-as-se-op",
            "bool", "false",
            "Flag which identifies whether an Interpolate operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// OptimizeSliceExpand
//

def OptimizeSliceExpand : PassBase<"optimize-slice-expand", "vpux::FunctionPass"> {
    let summary = "Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand in order to avoid extra DMAs
    }];

    let constructor = "vpux::IE::arch37xx::createOptimizeSliceExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateExpand
//

def PropagateExpand : PassBase<"propagate-expand", "vpux::FunctionPass"> {
    let summary = "Propagate Expand operation in order to fuse it with other layers";

    let description = [{
        Propagate Expand through Eltwise Add in case layers before might be fused with Expand
        in following cases:
        1. PermuteQuntize might be fused with Expand in FusePermuteQuantizeExpand pass
        2. DepthToSpace is used with padded channels' descriptor
        3. SpaceToDepth might be executed with expanded input on dpu with following convolution with padded filter
    }];

    let constructor = "vpux::IE::arch37xx::createPropagateExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePermuteQuantizeExpand
//

def FusePermuteQuantizeExpand : PassBase<"fuse-permute-quantize-expand", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute-Expand combination in 1 common operation";

    let description = [{
        Converts Quantize-MemPermute-Expand combination in 1 common operation.
    }];

    let constructor = "vpux::IE::arch37xx::createFusePermuteQuantizeExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandActivationChannels
//

def ExpandActivationChannels : PassBase<"expand-activation-channels", "vpux::FunctionPass"> {
    let summary = "Align input tensors shape of DPU operation with hardware requirements";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass processes operations, which can be compile as a DPU tasks and
            expands channels number to number divisible by 16 in case they doesn't satisfy hardware requirements.
        The input channels could be aligned to 4 instead of 16 for compressed convolutions.
    }];

    let constructor = "vpux::IE::arch37xx::createExpandActivationChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seTransposedConvEnabled", "se-transposed-conv-enabled",
            "bool", "false",
            "(Experimental) Flag to identify whether Transposed Convolutions can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// UnrollBatch
//

def UnrollBatch : PassBase<"unroll-batch", "vpux::FunctionPass"> {
    let summary = "Split inputs of NCE tasks when their batch size is greater than 1";

    let description = [{
        This pass splits inputs of NCE tasks by batch.

        For example:
        * `FullyConnected` input with 2x64 geometry will be split by two inputs with 1x64 dimensions.
        * `Convolution` input 3x16x32x64 will be split into three 1x16x32x64 inputs.
        Resulting tensors go to corresponding operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::arch37xx::createUnrollBatchPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertFFTToConv
//

def ConvertFFTToConv : PassBase<"convert-fft-to-conv", "vpux::FunctionPass"> {
    let summary = "Replace FFT operations (I/R/ DFT) with a subgraph of smaller operations";

    let description = [{
        Decomposes `FFT` operations (DFT, IDFT, RDFT, IRDFT) into smaller `convolution` friendly operations.
    }];

    let constructor = "vpux::IE::arch37xx::createConvertFFTToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMixPrecision
//

def ConvertToMixedPrecision: PassBase<"convert-to-mixed-precision", "vpux::FunctionPass"> {
    let summary = "Convert DPU task without fake quantize behind to mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        Converts DPU task to mixed-precision operation where there is no quantize operation for the output of a DPU task
    }];

    let constructor = "vpux::IE::arch37xx::createConvertToMixedPrecision()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableFloatInQuantWeightsMixedMode", "enable-float-in-quant-weights-mixed-mode",
            "bool", "true",
            "Enable mixed mode for NCE tasks with float input and quantized weights"
        >
    ];
}

//
// OptimizeNetworkInputConvert
//

def OptimizeNetworkInputConvert: PassBase<"optimize-network-input-convert", "vpux::FunctionPass"> {
    let summary = "Fuses outstanding Convert operations into input of quantized consumers";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        To avoid extra FP->Quant Convert operations at network start, fuses these Convert ops into quantized consumers
        resulting mixed precision operations with FP input and Quant output.
        As a side effect, also the original quantized weights are dequantized.
        There is later logic which attempts to enable mixed precision of FP input activations and Quant weights.
    }];

    let constructor = "vpux::IE::arch37xx::createOptimizeNetworkInputConvertPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
