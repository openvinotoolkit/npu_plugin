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

#ifndef VPUX_COMPILER_CORE_TYPE_INTERFACES
#define VPUX_COMPILER_CORE_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//
// NDTypeInterface
//

def NDTypeInterface : TypeInterface<"NDTypeInterface"> {
    let description = [{
        Interface for extracting shape, element type, layout, memory space and other tensor related properties from the type.
    }];

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Returns the shape of the underlying type",
            "vpux::ShapeRef", "getShape", (ins)
        >,

        InterfaceMethod<
            "Returns the memory shape of the underlying type",
            "vpux::MemShape", "getMemShape", (ins)
        >,

        InterfaceMethod<
            "Returns true if the underlying type is ranked",
            "bool", "hasRank", (ins)
        >,

        InterfaceMethod<
            "Returns the rank of the underlying type",
            "int64_t", "getRank", (ins)
        >,

        InterfaceMethod<
            "Returns the number of elements of the underlying type",
            "int64_t", "getNumElements", (ins)
        >,

        InterfaceMethod<
            "Returns the element type of the underlying type",
            "mlir::Type", "getElementType", (ins)
        >,

        InterfaceMethod<
            "Returns the layout of the underlying type",
            "vpux::DimsOrder", "getDimsOrder", (ins)
        >,

        InterfaceMethod<
            "Returns the memory space of the underlying type",
            "vpux::IndexedSymbolAttr", "getMemSpace", (ins)
        >,

        InterfaceMethod<
            "Returns the memory kind of the underlying type",
            "vpux::VPU::MemoryKind", "getMemoryKind", (ins)
        >,

        InterfaceMethod<
            "Returns the strides of the underlying type",
            "vpux::Strides", "getStrides", (ins)
        >,

        InterfaceMethod<
            "Returns the memory strides of the underlying type",
            "vpux::MemStrides", "getMemStrides", (ins)
        >,

        InterfaceMethod<
            "Returns the element type size of the underlying type",
            "vpux::Bit", "getElemTypeSize", (ins)
        >,

        InterfaceMethod<
            "Returns the total allocation size of the underlying type",
            "vpux::Byte", "getTotalAllocSize", (ins)
        >,

        InterfaceMethod<
            "Returns the compact allocation size of the underlying type",
            "vpux::Byte", "getCompactAllocSize", (ins)
        >,

       InterfaceMethod<
            "Returns the type with the new shape",
            "vpux::NDTypeInterface", "changeShape", (ins "vpux::ShapeRef":$shape)
        >,

        InterfaceMethod<
            "Returns the type with the new element type",
            "vpux::NDTypeInterface", "changeElemType", (ins "mlir::Type":$elemType)
        >,

        InterfaceMethod<
            "Returns the type with the new shape and element type",
            "vpux::NDTypeInterface", "changeShapeElemType", (ins "vpux::ShapeRef":$shape, "mlir::Type":$elemType)
        >,

        InterfaceMethod<
            "Returns the type with the new order",
            "vpux::NDTypeInterface", "changeDimsOrder", (ins "vpux::DimsOrder":$order)
        >,

        InterfaceMethod<
            "Returns the type with the new memory space",
            "vpux::NDTypeInterface", "changeMemSpace", (ins "vpux::IndexedSymbolAttr":$memSpace)
        >,

        InterfaceMethod<
            "Returns the type with the new strides",
            "vpux::NDTypeInterface", "changeStrides", (ins "vpux::StridesRef":$strides)
        >,

        InterfaceMethod<
            "Returns a tile from the type",
            "vpux::NDTypeInterface", "extractDenseTile", (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape)
        >,

        InterfaceMethod<
            "Returns a view tile from the type",
            "vpux::NDTypeInterface", "extractViewTile", (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape, "vpux::ShapeRef":$tileElemStrides)
        >,

        InterfaceMethod<
            "Returns the type without strides information",
            "vpux::NDTypeInterface", "eraseTiledInfo", (ins)
        >,

        InterfaceMethod<
            "Returns the padded type",
            "vpux::NDTypeInterface", "pad", (ins "vpux::ShapeRef":$padBefore, "vpux::ShapeRef":$padAfter)
        >,
    ];

    let extraClassDeclaration = [{
        template <typename Enum, typename OutT = vpux::NDTypeInterface>
        using nd_type_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        nd_type_if<Enum> changeMemSpace(Enum kind) const {
            return changeMemSpace(vpux::IndexedSymbolAttr::get(getContext(), stringifyEnum(kind)));
        }
    }];
}

#endif
