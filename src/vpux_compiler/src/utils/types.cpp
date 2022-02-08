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

#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4);
}

mlir::IntegerType vpux::getInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8);
}

mlir::IntegerType vpux::getInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16);
}

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64);
}

mlir::IntegerType vpux::getSInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getUInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getBoolType(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);
}

//
// TypeSize
//

Bit vpux::getElemTypeSize(mlir::Type type) {
    if (const auto shaped = type.dyn_cast<mlir::ShapedType>()) {
        return getElemTypeSize(shaped.getElementType());
    }

    if (type.isIntOrFloat()) {
        return Bit(type.getIntOrFloatBitWidth());
    }

    if (const auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return Bit(qType.getStorageTypeIntegralWidth());
    }

    VPUX_THROW("Can't get type size for '{0}'", type);
}

Byte vpux::getTotalSize(mlir::ShapedType type) {
    if (type.getRank() == 0) {
        return getElemTypeSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getTotalSize(type);
}

Byte vpux::getCompactSize(mlir::ShapedType type) {
    const auto typeSize = static_cast<Bit>(getElemTypeSize(type));
    if (type.getRank() == 0) {
        return typeSize;
    }

    const auto shape = getShape(type);
    return shape.totalSize() * typeSize;
}

Byte vpux::getCompactSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getCompactSize(type);
}

Optional<int32_t> vpux::getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape) {
    auto prevArray = to_small_vector(prevShape.toValues());
    auto newArray = to_small_vector(newShape.toValues());

    if (checked_cast<size_t>(axis) >= prevShape.size()) {
        return None;
    }

    const auto isEqualOne = [](int64_t val) {
        return val == 1;
    };

    auto firstPrevIter = std::find_if_not(prevArray.begin(), prevArray.end(), isEqualOne);
    auto firstNewIter = std::find_if_not(newArray.begin(), newArray.end(), isEqualOne);

    if (firstPrevIter == prevArray.end() || firstNewIter == newArray.end()) {
        return axis;
    }

    const auto gap = checked_cast<int32_t>(std::distance(newArray.begin(), firstNewIter) -
                                           std::distance(prevArray.begin(), firstPrevIter));
    const auto newArraySize = newArray.size();

    prevArray.erase(std::remove_if(prevArray.begin(), prevArray.end(), isEqualOne), prevArray.end());
    newArray.erase(std::remove_if(newArray.begin(), newArray.end(), isEqualOne), newArray.end());

    if (!std::equal(prevArray.begin(), prevArray.end(), newArray.begin(), newArray.end())) {
        return None;
    }

    if (axis < -gap || axis + gap >= checked_cast<int32_t>(newArraySize)) {
        return None;
    }

    return axis + gap;
}

//
// MemRefType utilities
//

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setLayout(mlir::AffineMapAttr::get(order.toAffineMap(elemType.getContext())));
    builder.setMemorySpace(memSpace);
    return builder;
}

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, StridesRef strides,
                                     IndexedSymbolAttr memSpace) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);
    VPUX_THROW_UNLESS(shape.size() == strides.size(), "Strides '{0}' doesn't match shape '{1}'", strides, shape);

    const auto elemSize = getElemTypeSize(elemType);

    const auto memShape = order.toMemoryOrder(shape);
    const auto memStrides = order.toMemoryOrder(strides);
    const auto compactReqs = StrideReqs::compact(shape.size());
    if (compactReqs.checkStrides(memStrides, elemSize, memShape)) {
        // Do not store compact strides.
        return getMemRefType(shape, elemType, order, memSpace);
    }

    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));

    auto* ctx = elemType.getContext();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layoutAttr = IERT::MemRefAttr::get(orderAttr, stridesAttr, ctx);

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setLayout(layoutAttr.cast<mlir::MemRefLayoutAttrInterface>());
    builder.setMemorySpace(memSpace);
    return builder;
}

mlir::MemRefType vpux::changeElemType(mlir::MemRefType origType, mlir::Type elemType) {
    const auto order = DimsOrder::fromType(origType);
    const auto shape = getShape(origType);
    const auto memSpace = getMemorySpace(origType);
    const auto newType = getMemRefType(shape, elemType, order, memSpace);

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::changeShape(mlir::MemRefType origType, ShapeRef shape) {
    const auto elemType = origType.getElementType();
    const auto memSpace = getMemorySpace(origType);

    const auto origOrder = DimsOrder::fromType(origType);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    const auto newType = getMemRefType(shape, elemType, newOrder, memSpace);

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::changeDimsOrder(mlir::MemRefType origType, DimsOrder order) {
    const auto shape = getShape(origType);
    const auto elemType = origType.getElementType();
    const auto memSpace = getMemorySpace(origType);
    return getMemRefType(shape, elemType, order, memSpace);
}

mlir::MemRefType vpux::changeMemSpace(mlir::MemRefType origType, IndexedSymbolAttr memSpace) {
    return mlir::MemRefType::Builder(origType).setMemorySpace(memSpace);
}

mlir::MemRefType vpux::getDenseTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape) {
    return eraseTiledInfo(getViewTileType(origType, tileOffsets, tileShape));
}

mlir::MemRefType vpux::getViewTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape,
                                       ShapeRef tileElemStrides) {
    const auto order = DimsOrder::fromType(origType);
    const auto memSpace = getMemorySpace(origType);

    auto tileElemType = origType.getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides(origType);
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto tileType = getMemRefType(tileShape, tileElemType, order, tileStrides, memSpace);

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, tileType).succeeded(), "Got invalid ShapedType '{0}'", tileType);

    return tileType;
}

mlir::MemRefType vpux::getPaddedType(mlir::MemRefType origType, ShapeRef padBefore, ShapeRef padAfter) {
    const auto order = DimsOrder::fromType(origType);
    const auto memSpace = getMemorySpace(origType);

    const auto origShape = getShape(origType);
    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(),
                      "Got non consistent 'padBefore' and 'padAfter' values in 'getPaddedType'");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(),
                      "Paddings and input shape are not consistent in 'getPaddedType'");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto newElemType = origType.getElementType();
    if (const auto perAxisQType = newElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        newElemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    const auto newType = getMemRefType(newShape, newElemType, order, memSpace);

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::eraseTiledInfo(mlir::MemRefType origType) {
    // Erase strides and offsets information from memory reference.

    const auto shape = getShape(origType);
    const auto elemType = origType.getElementType();
    const auto order = DimsOrder::fromType(origType);
    const auto memSpace = getMemorySpace(origType);
    return getMemRefType(shape, elemType, order, memSpace);
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order,
                                           IndexedSymbolAttr memSpace, bool sparse) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);

    const auto tensorDesc = IE::getTensorAttr(elemType.getContext(), order, memSpace, sparse);
    const auto newType = mlir::RankedTensorType::get(shape.raw(), elemType, tensorDesc);

    const auto loc = mlir::UnknownLoc::get(elemType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeElemType(mlir::RankedTensorType origType, mlir::Type elemType) {
    const auto newType = getTensorType(getShape(origType), elemType, DimsOrder::fromType(origType),
                                       IE::getMemorySpace(origType), IE::isSparse(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeShape(mlir::RankedTensorType origType, ShapeRef shape) {
    const auto origOrder = DimsOrder::fromType(origType);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    auto elemType = origType.getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(origType), shape);
        if (axis.hasValue()) {
            elemType = changeAxis(perAxisType, axis.getValue());
        }
    }
    const auto newType = getTensorType(shape, elemType, newOrder, IE::getMemorySpace(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeDimsOrder(mlir::RankedTensorType origType, DimsOrder order) {
    return getTensorType(getShape(origType), origType.getElementType(), order, IE::getMemorySpace(origType),
                         IE::isSparse(origType));
}

mlir::RankedTensorType vpux::changeSparse(mlir::RankedTensorType origType, bool sparse) {
    return getTensorType(getShape(origType), origType.getElementType(), DimsOrder::fromType(origType),
                         IE::getMemorySpace(origType), sparse);
}

mlir::RankedTensorType vpux::changeMemSpace(mlir::RankedTensorType origType, IndexedSymbolAttr memSpace) {
    return getTensorType(getShape(origType), origType.getElementType(), DimsOrder::fromType(origType), memSpace,
                         IE::isSparse(origType));
}

mlir::RankedTensorType vpux::getDenseTileType(mlir::RankedTensorType origType, ShapeRef tileOffsets,
                                              ShapeRef tileShape) {
    auto elemType = origType.getElementType();
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto newType = getTensorType(tileShape, elemType, DimsOrder::fromType(origType), IE::getMemorySpace(origType),
                                       IE::isSparse(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::getPaddedType(mlir::RankedTensorType origType, ShapeRef padBefore, ShapeRef padAfter) {
    const auto origShape = getShape(origType);

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(),
                      "Got non consistent 'padBefore' and 'padAfter' values in 'getPaddedType'");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(),
                      "Paddings and input shape are not consistent in 'getPaddedType'");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto elemType = origType.getElementType();
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    const auto newType = getTensorType(newShape, elemType, DimsOrder::fromType(origType), IE::getMemorySpace(origType),
                                       IE::isSparse(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

IndexedSymbolAttr vpux::getMemorySpace(mlir::MemRefType type) {
    auto memSpaceAttr = type.getMemorySpace();
    if (memSpaceAttr == nullptr) {
        return nullptr;
    }

    auto memSpace = memSpaceAttr.dyn_cast<IndexedSymbolAttr>();
    VPUX_THROW_UNLESS(memSpace != nullptr, "Unsupported memory space attribute'{0}'", memSpaceAttr);

    return memSpace;
}

//
// ShapedType utilities
//

mlir::ShapedType vpux::changeElemType(mlir::ShapedType origType, mlir::Type elemType) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return changeElemType(memref, elemType);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeElemType(tensor, elemType);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::changeShape(mlir::ShapedType origType, ShapeRef shape) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return changeShape(memref, shape);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeShape(tensor, shape);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::changeDimsOrder(mlir::ShapedType origType, DimsOrder order) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return changeDimsOrder(memref, order);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeDimsOrder(tensor, order);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::changeMemSpace(mlir::ShapedType origType, IndexedSymbolAttr memSpace) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return changeMemSpace(memref, memSpace);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeMemSpace(tensor, memSpace);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::changeSparse(mlir::ShapedType origType, bool sparse) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeSparse(tensor, sparse);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::getDenseTileType(mlir::ShapedType origType, ShapeRef tileOffsets, ShapeRef tileShape) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return getDenseTileType(memref, tileOffsets, tileShape);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return getDenseTileType(tensor, tileOffsets, tileShape);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

mlir::ShapedType vpux::getPaddedType(mlir::ShapedType origType, ShapeRef padBefore, ShapeRef padAfter) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memref) {
                return getPaddedType(memref, padBefore, padAfter);
            })
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return getPaddedType(tensor, padBefore, padAfter);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}
