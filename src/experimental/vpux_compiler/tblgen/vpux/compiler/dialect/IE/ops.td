//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS
#define VPUX_COMPILER_DIALECT_IE_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/dialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class IE_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            IE_Dialect,
            mnemonic,
            traits
        >;

class IE_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        IE_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    NoSideEffect,
                    NativeOpTrait<"InferTensorType">,
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<InferShapedTypeOpInterface>,
                    DeclareOpInterfaceMethods<LayerInterface>
                ]
            )
        >;

//
// CNNNetworkOp
//

def IE_CNNNetworkOp :
        IE_Op<
            "CNNNetwork",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                SingleBlockImplicitTerminator<"vpux::IE::EndOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<NetInfoInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
        > {
    let summary = "InferenceEngine CNN Network description";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about InferenceEngine CNN Network:

          * Precision and layout for user-provided inputs.
          * Precision and layout for user-provided outputs.
          * Entry point (Function name) for the network inference.
    }];

    let arguments = (ins
        StrAttr:$netName,
        FlatSymbolRefAttr:$entryPoint
    );

    let regions = (region
        SizedRegion<1>:$inputsInfo,
        SizedRegion<1>:$outputsInfo
    );

    let assemblyFormat = [{
        attr-dict
        $netName `at` $entryPoint
        `inputsInfo` `:` $inputsInfo
        `outputsInfo` `:` $outputsInfo
    }];
}

def IE_DataInfoOp :
        IE_Op<
            "DataInfo",
            [
                IsolatedFromAbove,
                HasParent<"vpux::IE::CNNNetworkOp">,
                DeclareOpInterfaceMethods<DataInfoInterface>
            ]
        > {
    let summary = "Information about InferenceEngine CNN Network input/output Data object";

    let description = [{
        This operation is bound to `IE.CNNNetwork` Operation and holds information about Data object:

          * Name
          * User-defined precision (element type)
          * User-defined layout
    }];

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$precision,
        IE_Layout:$layout
    );

    let assemblyFormat = [{
        attr-dict
        $name `,` $precision `,` $layout
    }];
}

def IE_EndOp :
        IE_Op<
            "End",
            [
                NoSideEffect,
                Terminator,
                HasParent<"vpux::IE::CNNNetworkOp">
            ]
        > {
    let summary = "End indicator for IE CNNNetwork inputs/outputs info lists";

    let assemblyFormat = "attr-dict";
}

//
// ConvertOp
//

def IE_ConvertOp :
        IE_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "InferenceEngine Convert layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// SoftMaxOp
//

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine SoftMax layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I32Attr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasCanonicalizer = 1;
}

#endif
