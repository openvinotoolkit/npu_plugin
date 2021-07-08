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

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_CONST_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// TransformAttrInterface
//

def Const_TransformAttrInterface : AttrInterface<"TransformAttrInterface"> {
    let description = [{
        The interface for Attributes, which holds information about lazy constant folding operation.
    }];

    let cppNamespace = "vpux::Const";

    let methods = [
        InterfaceMethod<
            "Infer output type",
            "mlir::ShapedType", "inferOutputType",
            (ins "mlir::ShapedType":$input)
        >,

        InterfaceMethod<
            "Transform the constant content",
            "vpux::Const::Content", "transform",
            // HACK: it should be rvalue-reference, but can't use it due to auto-generated code limitation
            (ins "vpux::Const::Content&":$input)
        >
    ];
}

#endif
