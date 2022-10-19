//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_CONST_OPS
#define VPUX_COMPILER_DIALECT_CONST_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/ELF/ops_interfaces.td"


include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//
// Base classes
//

class Const_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            Const_Dialect,
            mnemonic,
            !listconcat(
              traits,
              [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
              ]
            )
        >;

//
// DeclareOp
//

def Const_DeclareOp :
        Const_Op<
            "Declare",
            [
                ConstantLike,
                NoSideEffect,
                DeclarationOp,
                DotInterface,
                OpAsmOpInterface
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

    let hasFolder = 1;

    let verifier = [{
        return verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUAMARINE;
        }

        void getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
            setNameFn(getResult(), "cst");
        }
    }];

    let assemblyFormat = [{
        attr-dict type($output) `=` $content
    }];
}

#endif
