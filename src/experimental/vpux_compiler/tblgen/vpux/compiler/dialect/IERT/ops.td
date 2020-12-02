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

#ifndef VPUX_COMPILER_DIALECT_IERT_OPS
#define VPUX_COMPILER_DIALECT_IERT_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IERT/dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class IERT_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            IERT_Dialect,
            mnemonic,
            traits
        >;

class IERT_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        IERT_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                    DeclareOpInterfaceMethods<LayerInterface>
                ]
            )
        > {
    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands)
    }];
}

//
// CNNNetworkOp
//

def IERT_CNNNetworkOp :
        IERT_Op<
            "CNNNetwork",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                SingleBlockImplicitTerminator<"vpux::IERT::EndOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<NetInfoInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
        > {
    let summary = "InferenceEngine CNN Network run-time description";

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

def IERT_DataInfoOp :
        IERT_Op<
            "DataInfo",
            [
                IsolatedFromAbove,
                HasParent<"vpux::IERT::CNNNetworkOp">,
                DeclareOpInterfaceMethods<DataInfoInterface>
            ]
        > {
    let summary = "Information about InferenceEngine CNN Network input/output Data object";

    let description = [{
        This operation is bound to `IERT.CNNNetwork` Operation and holds information about Data object:

          * Name
          * User-defined precision (element type)
          * User-defined layout
    }];

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$precision,
        AffineMapAttr:$layout
    );

    let assemblyFormat = [{
        attr-dict
        $name `,` $precision `,` $layout
    }];
}

def IERT_EndOp :
        IERT_Op<
            "End",
            [
                NoSideEffect,
                Terminator,
                HasParent<"vpux::IERT::CNNNetworkOp">
            ]
        > {
    let summary = "End indicator for IERT CNNNetwork inputs/outputs info lists";

    let assemblyFormat = "attr-dict";
}

//
// ConvertOp
//

def IERT_ConvertOp :
        IERT_LayerOp<
            "Convert",
            [
                SameOperandsShape,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Convert layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output
    );
}

//
// SoftMaxOp
//

def IERT_SoftMaxOp :
        IERT_LayerOp<
            "SoftMax",
            [
                SameOperandsShape,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time SoftMax layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32Attr:$axisInd
    );
}

#endif
