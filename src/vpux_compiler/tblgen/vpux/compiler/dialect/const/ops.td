//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_CONST_OPS
#define VPUX_COMPILER_DIALECT_CONST_OPS

include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attributes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

//
// Base classes
//

class Const_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            Const_Dialect,
            mnemonic,
            traits
        >;

//
// DeclareOp
//

def Const_DeclareOp :
        Const_Op<
            "Declare",
            [
                ConstantLike,
                NoSideEffect
            ]
        > {
    let summary = "Constant tensor/buffer declaration";

    let description = [{
        This operation can perform extra lazy constant folding transformations for constant content.
    }];

    let arguments = (ins
        Const_ContentAttr:$content
    );

    let results = (outs
        AnyTypeOf<[AnyStaticShapeTensor, AnyStaticShapeMemRef]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let verifier = [{
        return verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict type($output) `=` $content
    }];
}

#endif
