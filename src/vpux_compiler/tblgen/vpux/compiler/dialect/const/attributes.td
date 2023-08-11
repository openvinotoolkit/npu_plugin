//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES

include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attr_interfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//
// Base classes
//

class Const_Attr<string name, list<Trait> traits = [], string baseCppClass = "mlir::Attribute"> :
        AttrDef<Const_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

//
// OpaqueElementsAttr
//

def Const_OpaqueElementsAttr : Const_Attr<"OpaqueElements", [ElementsAttrInterface]> {
    let summary = "An opaque representation of a multi-dimensional array";
    let description = [{
        An opaque elements attribute is an elements attribute where the content of
        the value is opaque. Attribute is used to store a reference to shared data 
        received by compiler frontend. (It does not own the memory)

        Note: The parsed string literal must be in hexadecimal form.

        Examples:

        ```mlir
        opaque<"elided_large_const", "0xDEADBEEF"> : tensor<10xi32>
        ```
    }];

    let parameters = (ins AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
                          StringRefParameter<"">:$value);
    let builders = [
        AttrBuilderWithInferredContext<(ins "mlir::ShapedType":$type,
                                          "mlir::StringRef":$value), [{
            return $_get(type.getContext(), type, value);
        }]>,
    ];

    let extraClassDeclaration = [{
        using ValueType = StringRef;
    }];

    let storageNamespace = "details";
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
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
                return vpux::Const::ContentAttr::addTransformation(input, transformation);
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
    let hasCustomAssemblyFormat = 1;

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
        vpux::Const::ContentAttr relocateWeightsTablePointers(ArrayRef<int32_t> weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets,
                                                              uint64_t weightsElemByteSize, VPUIP::CompressionSchemeAttr weightsCompression) const;
        vpux::Const::ContentAttr swizzleConstant(uint64_t swizzleKey, uint64_t arch) const;
        vpux::Const::ContentAttr add(double bias) const;
        vpux::Const::ContentAttr reshape(vpux::ShapeRef newShape) const;
        vpux::Const::ContentAttr reverse(Dim axis) const;
        vpux::Const::ContentAttr reorder(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
        vpux::Const::ContentAttr subview(vpux::ShapeRef offset, vpux::ShapeRef shape) const;
        vpux::Const::ContentAttr bitPack(int64_t width) const;
        vpux::Const::ContentAttr transpose(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr expandDilated(vpux::ShapeRef dilations) const;
        vpux::Const::ContentAttr getSparsityMap() const;
        vpux::Const::ContentAttr sparsify(bool compressOutputType) const;

        static vpux::Const::ContentAttr addTransformation(vpux::Const::ContentAttr input, vpux::Const::TransformAttrInterface transformation);
        int64_t getNumberOfElements() const;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
}

//
// ReverseAttr
//

def Const_ReverseAttr : Const_Attr<"Reverse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Reverse constant content on specified axis";

    let description = [{
        This transformation reverses content on a specified axis.
        For example, considering an input with dimensions (d0, d1, d2, d3),
        reversing on axis d1 will result in the data of every subpart (d2, d3)
        being reversed.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$axis
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis), [{
                return $_get(axis.getContext(), axis);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
}

//
// BitPackAttr
//

def Const_BitPackAttr : Const_Attr<"BitPack",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>,
         DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
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
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
}

//
// WeightsTablePtrs
//

def Const_RelocateWeightsTableAttr : Const_Attr<"RelocateWeightsTable",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>, DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Patches offsets in the weights table";

    let parameters = (ins
        "mlir::ArrayAttr":$weightsPtr,
        "mlir::IntegerAttr":$sparsityPtr,
        "mlir::ArrayAttr":$offsets,
        "mlir::IntegerAttr":$weightsElemByteSize,
        "VPUIP::CompressionSchemeAttr":$weightsCompression
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$weightsPtr, "mlir::IntegerAttr":$sparsityPtr, "mlir::ArrayAttr":$offsets,
                 "mlir::IntegerAttr":$weightsElemByteSize, "VPUIP::CompressionSchemeAttr":$weightsCompression), [{
                return $_get(weightsPtr.getContext(), weightsPtr, sparsityPtr, offsets, weightsElemByteSize, weightsCompression);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SwizzleConstant
//

def Const_SwizzleConstantAttr : Const_Attr<"SwizzleConstant",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>,
        DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let summary = "Swizzles the constant based on swizzle key";

    let description = [{
        This transformation applies swizzling transformation to a constant content.
        Configuration of transformation depends on architecture and swizzleKey.
        Constant content will be size aligned (alignSize flag) if its default size
        does not meet HW requirements for this transformation.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$swizzleKey,
        "mlir::IntegerAttr":$arch,
        "mlir::BoolAttr":$alignSize
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$swizzleKey, "mlir::IntegerAttr": $arch), [{
                return $_get(swizzleKey.getContext(), swizzleKey, arch, nullptr);
            }]
        >,
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$swizzleKey, "mlir::IntegerAttr": $arch, "mlir::BoolAttr": $alignSize), [{
                return $_get(swizzleKey.getContext(), swizzleKey, arch, alignSize);
            }]
        >,
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
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
    let hasCustomAssemblyFormat = 1;
}

//
// GetSparsityMapAttr
//

def Const_GetSparsityMapAttr : Const_Attr<"GetSparsityMap",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
    ]> {
    let summary = "Generate sparsity map";

    let storageNamespace = "details";
}

//
// Sparsify
//

def Const_SparsifyAttr : Const_Attr<"Sparsify", [
        DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement", "updateAttributes"]>
    ]> {
    let summary = "Remove zeros / zero-points from constant";

    let parameters = (ins
        "mlir::BoolAttr":$compressOutputType,
        "mlir::ElementsAttr":$numActualElements
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType), [{
                return $_get(compressOutputType.getContext(), compressOutputType, nullptr);
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType, "mlir::ElementsAttr":$numActualElements), [{
                return $_get(compressOutputType.getContext(), compressOutputType, numActualElements);
            }]
        >
    ];

    let extraClassDeclaration = [{
        static int64_t countNumTotalElements(vpux::Const::SparsifyAttr transformation);
    }];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

#endif
