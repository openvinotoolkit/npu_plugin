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

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES

include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attr_interfaces.td"

include "mlir/IR/SubElementInterfaces.td"

//
// Base classes
//

class Const_Attr<string name, list<Trait> traits = [], string baseCppClass = "mlir::Attribute"> :
        AttrDef<Const_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

//
// ContentAttr
//

def Const_ContentAttr : Const_Attr<"Content", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Lazy folded constant content";

    let description = [{
        This attribute holds base constant and transformation applied to it.
        It provides and API to get transformed values on the fly.
    }];

    let parameters = (ins
        "mlir::ElementsAttr":$baseContent,
        "mlir::ArrayAttr":$transformations,
        "vpux::NDTypeInterface":$finalType
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent), [{
                return $_get(baseContent.getContext(), baseContent, mlir::ArrayAttr(), baseContent.getType().cast<vpux::NDTypeInterface>());
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "vpux::Const::ContentAttr":$input, "vpux::Const::TransformAttrInterface":$transformation), [{
                auto transformations = input.getTransformations();
                transformations.push_back(transformation);

                const auto transformationsRaw = to_small_vector(
                    transformations | transformed([](vpux::Const::TransformAttrInterface attr) -> mlir::Attribute {
                        return attr;
                    }));

                const auto transformationsAttr = mlir::ArrayAttr::get(input.getContext(), transformationsRaw);

                return $_get(input.getContext(), input.getBaseContent(), transformationsAttr,
                             transformation.inferOutputType(input.getType()));
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent, "mlir::ArrayAttr":$transformations, "vpux::NDTypeInterface":$finalType), [{
                return $_get(baseContent.getContext(), baseContent, transformations, finalType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let genAccessors = 0;

    let extraClassDeclaration = [{
        using ValueType = vpux::Const::Content;

        ValueType fold() const;

        mlir::ElementsAttr getBaseContent() const;
        vpux::SmallVector<vpux::Const::TransformAttrInterface> getTransformations() const;
        vpux::NDTypeInterface getType() const;

        vpux::Const::ContentAttr broadcast(Dim axis, int64_t value) const;
        vpux::Const::ContentAttr convertElemType(mlir::Type newElemType) const;
        vpux::Const::ContentAttr quantCast(mlir::quant::QuantizedType newElemType = nullptr) const;
        vpux::Const::ContentAttr dequantize() const;
        vpux::Const::ContentAttr rescale(double scale) const;
        vpux::Const::ContentAttr relocateWeightsTablePointers(uint64_t weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets) const;
        vpux::Const::ContentAttr add(double bias) const;
        vpux::Const::ContentAttr reshape(vpux::ShapeRef newShape) const;
        vpux::Const::ContentAttr reorder(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
        vpux::Const::ContentAttr subview(vpux::ShapeRef offset, vpux::ShapeRef shape) const;
        vpux::Const::ContentAttr bitPack(int64_t width) const;
        vpux::Const::ContentAttr transpose(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr expandDilated(vpux::ShapeRef dilations) const;
    }];

    let storageType = "vpux::Const::ContentAttr";
    let returnType = "vpux::Const::Content";
    let convertFromStorage = "$_self.fold()";
}

//
// ConvertElemTypeAttr
//

def Const_ConvertElemTypeAttr : Const_Attr<"ConvertElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Convert constant content element type";

    let parameters = (ins
        "mlir::Type":$elemType
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::Type":$elemType), [{
                return $_get(elemType.getContext(), elemType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// QuantCastAttr
//

def Const_QuantCastAttr : Const_Attr<"QuantCast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Cast element type from raw integer to quantized type";

    let parameters = (ins
        "mlir::quant::QuantizedType":$elemType
    );

    let builders = [

        AttrBuilder<
            (ins "mlir::quant::QuantizedType":$elemType), [{
                return $_get($_ctxt, elemType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// DequantizeAttr
//

def Const_DequantizeAttr : Const_Attr<"Dequantize",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Dequantize constant content";
}

//
// RescaleAttr
//

def Const_RescaleAttr : Const_Attr<"Rescale",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Rescale constant content";

    let parameters = (ins
        "mlir::FloatAttr":$scale
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$scale), [{
                return $_get(scale.getContext(), scale);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
}

//
// AddAttr
//

def Const_AddAttr : Const_Attr<"Add",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Add constant content";

    let parameters = (ins
        "mlir::FloatAttr":$bias
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$bias), [{
                return $_get(bias.getContext(), bias);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
}

//
// ReshapeAttr
//

def Const_ReshapeAttr : Const_Attr<"Reshape",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Reshape constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$shape), [{
                return $_get(shape.getContext(), shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// ReorderAttr
//

def Const_ReorderAttr : Const_Attr<"Reorder",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Reorder constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// PadWithZeroAttr
//

def Const_PadWithZeroAttr : Const_Attr<"PadWithZero",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Pad constant content with zeros";

    let parameters = (ins
        "mlir::ArrayAttr":$padBefore,
        "mlir::ArrayAttr":$padAfter
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$padBefore, "mlir::ArrayAttr":$padAfter), [{
                return $_get(padBefore.getContext(), padBefore, padAfter);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// SubViewAttr
//

def Const_SubViewAttr : Const_Attr<"SubView",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Extract subview from constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$offset,
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$offset, "mlir::ArrayAttr":$shape), [{
                return $_get(offset.getContext(), offset, shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// BroadcastAttr
//

def Const_BroadcastAttr : Const_Attr<"Broadcast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Broadcast axis by value of constant content";

    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        "mlir::IntegerAttr":$value
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis, "mlir::IntegerAttr":$value), [{
                return $_get(value.getContext(), axis, value);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
}

//
// BitPackAttr
//

def Const_BitPackAttr : Const_Attr<"BitPack",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Pack sub-byte constant content";

    let parameters = (ins
        "mlir::IntegerAttr":$width
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$width), [{
                return $_get(width.getContext(), width);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// TransposeAttr
//

def Const_TransposeAttr : Const_Attr<"Transpose",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Transpose constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

//
// WeightsTablePtrs
//

def Const_RelocateWeightsTableAttr : Const_Attr<"RelocateWeightsTable",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Patches offsets in the weights table";

    let parameters = (ins
        "mlir::IntegerAttr":$weightsPtr,
        "mlir::IntegerAttr":$sparsityPtr,
        "mlir::ArrayAttr":$offsets
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$weightsPtr, "mlir::IntegerAttr":$sparsityPtr, "mlir::ArrayAttr":$offsets), [{
                return $_get(weightsPtr.getContext(), weightsPtr, sparsityPtr, offsets);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
}

//
// ExpandDilatedAttr
//

def Const_ExpandDilatedAttr : Const_Attr<"ExpandDilated",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Expand constant content with zeros according to dilations";

    let parameters = (ins
        "mlir::ArrayAttr":$dilations
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$dilations), [{
                return $_get(dilations.getContext(), dilations);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
}

#endif
