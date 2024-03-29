//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
            "vpux::NDTypeInterface", "inferOutputType",
            (ins "vpux::NDTypeInterface":$input)
        >,

        InterfaceMethod<
            "Transform the constant content",
            "vpux::Const::Content", "transform",
            // HACK: it should be rvalue-reference, but can't use it due to auto-generated code limitation
            (ins "vpux::Const::Content&":$input)
        >,

        InterfaceMethod<
            "Get transformation name",
            "llvm::StringLiteral", "getTransformationName",
            (ins), [{}], [{
                return $_attr.getMnemonic();
            }]
        >,

        InterfaceMethod<
            "Get transformation location requirement",
            "vpux::Const::details::PositionRequirement", "getPositionRequirement",
            (ins), [{}], [{
                return vpux::Const::details::PositionRequirement::NONE;
            }]
        >,

        InterfaceMethod<
            "Update attributes of transformation and return new one",
            "Const::TransformAttrInterface",
            "updateAttributes",
            (ins "mlir::ElementsAttr&":$baseContent, "ArrayRef<mlir::Attribute>":$prevTransformations), [{}], [{
                std::ignore = prevTransformations;
                std::ignore = baseContent;
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Supports sub byte storage type",
            "bool", "supportsSubByteStorageType",
            (ins), [{}], [{
                return false;
            }]
        >
    ];
}

//
// RequestQueueAttrInterface
//

def Const_RequestQueueAttrInterface : AttrInterface<"RequestQueueAttrInterface"> {
    let description = [{
        The interface for Attributes which are used as elements in the request queue for folding in background.
    }];

    let cppNamespace = "vpux::Const";
}

#endif
