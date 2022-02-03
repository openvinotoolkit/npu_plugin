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

#include "vpux/compiler/core/type_interfaces.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

//
// Generated
//

#include <vpux/compiler/core/generated/type_interfaces.cpp.inc>

//
// TensorPropertiesTypeInterface
//

vpux::ShapeRef TensorPropertiesTypeInterface::getShape(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::ShapeRef>(type)
            .Case<mlir::RankedTensorType, mlir::UnrankedTensorType>([](auto tensor) {
                return vpux::ShapeRef(tensor.getShape());
            })
            .Default([](mlir::Type type) -> vpux::ShapeRef {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::MemShape TensorPropertiesTypeInterface::getMemShape(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemShape'. Got '{0}'", type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool TensorPropertiesTypeInterface::hasRank(mlir::Type type) const {
    return type.isa<mlir::RankedTensorType>();
}

int64_t TensorPropertiesTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return tensor.getRank();
}

int64_t TensorPropertiesTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getNumElements'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return tensor.getNumElements();
}

mlir::Type TensorPropertiesTypeInterface::getElementType(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::RankedTensorType, mlir::UnrankedTensorType>([](auto tensor) {
                return tensor.getElementType();
            })
            .Default([](mlir::Type type) -> mlir::Type {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::DimsOrder TensorPropertiesTypeInterface::getDimsOrder(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getDimsOrder'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return DimsOrder::fromAffineMap(IE::getOrder(tensor));
}

vpux::IndexedSymbolAttr TensorPropertiesTypeInterface::getMemSpace(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemSpace'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return IE::getMemorySpace(tensor);
}

vpux::VPU::MemoryKind TensorPropertiesTypeInterface::getMemoryKind(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemoryKind'. Got '{0}'", type);
    const auto memSpace = getMemSpace(type);

    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides TensorPropertiesTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getStrides'. Got '{0}'", type);
    const auto memStrides = getMemStrides(type);
    const auto order = getDimsOrder(type);
    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides TensorPropertiesTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemStrides'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    const auto order = getDimsOrder(type);
    // Tensors are always compact
    return StrideReqs::compact(order.numDims()).calcStrides(order, tensor);
}

vpux::Bit TensorPropertiesTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte TensorPropertiesTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getTotalAllocSize'. Got '{0}'", type);
    if (getRank(type) == 0) {
        return getElemTypeSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

vpux::Byte TensorPropertiesTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getCompactAllocSize'. Got '{0}'", type);
    const auto typeSize = static_cast<Bit>(getElemTypeSize(type));
    if (getRank(type) == 0) {
        return typeSize;
    }

    const auto shape = getShape(type);
    return shape.totalSize() * typeSize;
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::changeShape(mlir::Type type,
                                                                               vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeShape'. Got '{0}'", type);

    const auto tensor = type.cast<mlir::RankedTensorType>();
    const auto origOrder = getDimsOrder(type);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    const auto newType =
            vpux::getTensorType(shape, getElementType(type), newOrder, getMemSpace(type), IE::isSparse(tensor));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::changeElemType(mlir::Type type,
                                                                                  mlir::Type elemType) const {
    auto newType = llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                           .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                               return vpux::getTensorType(getShape(type), elemType, getDimsOrder(type),
                                                          getMemSpace(type), IE::isSparse(tensor));
                           })
                           .Case<mlir::UnrankedTensorType>([&](mlir::UnrankedTensorType) {
                               return mlir::UnrankedTensorType::get(elemType);
                           })
                           .Default([](mlir::Type type) -> mlir::ShapedType {
                               VPUX_THROW("Unsupported type '{0}'", type);
                           });

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::changeDimsOrder(mlir::Type type,
                                                                                   vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeDimsOrder'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return vpux::getTensorType(getShape(type), getElementType(type), order, getMemSpace(type), IE::isSparse(tensor));
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::changeMemSpace(
        mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeMemSpace'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return vpux::getTensorType(getShape(type), getElementType(type), getDimsOrder(type), memSpace,
                               IE::isSparse(tensor));
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::extractDenseTile(mlir::Type type,
                                                                                    vpux::ShapeRef tileOffsets,
                                                                                    vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'extractDenseTile'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    auto elemType = getElementType(type);
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto newType =
            vpux::getTensorType(tileShape, elemType, getDimsOrder(type), getMemSpace(type), IE::isSparse(tensor));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::ShapedPropertiesTypeInterface TensorPropertiesTypeInterface::pad(mlir::Type type, vpux::ShapeRef padBefore,
                                                                       vpux::ShapeRef padAfter) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(), "Only RankedTensorType is supported for 'pad'. Got '{0}'",
                      type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    const auto origShape = getShape(type);

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(), "Got non consistent 'padBefore' and 'padAfter' values");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(), "Paddings and input shape are not consistent");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto elemType = getElementType(type);
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    const auto newType =
            vpux::getTensorType(newShape, elemType, getDimsOrder(type), getMemSpace(type), IE::isSparse(tensor));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

//
// MemRefPropertiesTypeInterface
//

vpux::ShapeRef MemRefPropertiesTypeInterface::getShape(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::ShapeRef>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                return vpux::ShapeRef(memref.getShape());
            })
            .Default([](mlir::Type type) -> vpux::ShapeRef {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::MemShape MemRefPropertiesTypeInterface::getMemShape(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getMemShape'. Got '{0}'", type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool MemRefPropertiesTypeInterface::hasRank(mlir::Type type) const {
    return type.isa<mlir::MemRefType>();
}

int64_t MemRefPropertiesTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto memref = type.cast<mlir::MemRefType>();
    return memref.getRank();
}

int64_t MemRefPropertiesTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getNumElements'. Got '{0}'",
                      type);
    const auto memref = type.cast<mlir::MemRefType>();
    return memref.getNumElements();
}

mlir::Type MemRefPropertiesTypeInterface::getElementType(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                return memref.getElementType();
            })
            .Default([](mlir::Type type) -> mlir::Type {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::DimsOrder MemRefPropertiesTypeInterface::getDimsOrder(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getDimsOrder'. Got '{0}'", type);
    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }
    if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }
    VPUX_THROW("Missing layout information");
}

vpux::IndexedSymbolAttr MemRefPropertiesTypeInterface::getMemSpace(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::IndexedSymbolAttr>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                const auto memSpaceAttr = memref.getMemorySpace();
                if (memSpaceAttr == nullptr) {
                    return vpux::IndexedSymbolAttr();
                }

                auto memSpace = memSpaceAttr.template dyn_cast<vpux::IndexedSymbolAttr>();
                VPUX_THROW_UNLESS(memSpace != nullptr, "Unsupported memory space attribute'{0}'", memSpaceAttr);

                return memSpace;
            })
            .Default([](mlir::Type type) -> vpux::IndexedSymbolAttr {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::VPU::MemoryKind MemRefPropertiesTypeInterface::getMemoryKind(mlir::Type type) const {
    const auto memSpace = getMemSpace(type);

    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides MemRefPropertiesTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getStrides'. Got '{0}'", type);

    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();

    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);

        // Missing strides specification means compact strides.
        const auto order = getDimsOrder(type);
        const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(order, memref);

        return order.toLogicalOrder(memStrides);
    }

    if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        const auto elemStrides = parseIntArrayAttr<int64_t>(descAttr.strides());
        const auto elemSize = getElemTypeSize(type);

        return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                           return stride * elemSize;
                                       })));
    }

    VPUX_THROW("Unsupported MemRefType layout '{0}'", layout);
}

vpux::MemStrides MemRefPropertiesTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getMemStrides'. Got '{0}'",
                      type);
    const auto order = getDimsOrder(type);
    const auto strides = getStrides(type);
    return order.toMemoryOrder(strides);
}

vpux::Bit MemRefPropertiesTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte MemRefPropertiesTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getTotalAllocSize'. Got '{0}'",
                      type);
    if (getRank(type) == 0) {
        return getElemTypeSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

vpux::Byte MemRefPropertiesTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getCompactAllocSize'. Got '{0}'",
                      type);
    const auto typeSize = static_cast<Bit>(getElemTypeSize(type));
    if (getRank(type) == 0) {
        return typeSize;
    }

    const auto shape = getShape(type);
    return shape.totalSize() * typeSize;
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::changeShape(mlir::Type type,
                                                                               vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeShape'. Got '{0}'", type);

    const auto elemType = getElementType(type);
    const auto memSpace = getMemSpace(type);

    const auto origOrder = getDimsOrder(type);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    const auto newType = vpux::getMemRefType(shape, elemType, newOrder, memSpace);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::changeElemType(mlir::Type type,
                                                                                  mlir::Type elemType) const {
    auto newType =
            llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                    .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                        return vpux::getMemRefType(getShape(type), elemType, getDimsOrder(type), getMemSpace(type));
                    })
                    .Case<mlir::UnrankedMemRefType>([&](mlir::UnrankedMemRefType) {
                        return mlir::UnrankedMemRefType::get(elemType, getMemSpace(type));
                    })
                    .Default([](mlir::Type type) -> mlir::ShapedType {
                        VPUX_THROW("Unsupported type '{0}'", type);
                    });

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::changeDimsOrder(mlir::Type type,
                                                                                   vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeDimsOrder'. Got '{0}'",
                      type);
    return vpux::getMemRefType(getShape(type), getElementType(type), order, getMemSpace(type));
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::changeMemSpace(
        mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    return llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), getStrides(type),
                                           memSpace);
            })
            .Case<mlir::UnrankedMemRefType>([&](mlir::UnrankedMemRefType) {
                return mlir::UnrankedMemRefType::get(getElementType(type), memSpace);
            })
            .Default([](mlir::Type type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::extractDenseTile(mlir::Type type,
                                                                                    vpux::ShapeRef tileOffsets,
                                                                                    vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'extractDenseTile'. Got '{0}'",
                      type);
    const auto memref = type.cast<mlir::MemRefType>();
    return vpux::eraseTiledInfo(vpux::getViewTileType(memref, tileOffsets, tileShape));
}

vpux::ShapedPropertiesTypeInterface MemRefPropertiesTypeInterface::pad(mlir::Type type, vpux::ShapeRef padBefore,
                                                                       vpux::ShapeRef padAfter) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'pad'. Got '{0}'", type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);

    const auto origShape = getShape(type);
    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(), "Got non consistent 'padBefore' and 'padAfter' values");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(), "Paddings and input shape are not consistent");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto newElemType = getElementType(type);
    if (const auto perAxisQType = newElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        newElemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    const auto newType = getMemRefType(newShape, newElemType, order, memSpace);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}
