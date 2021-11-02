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

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
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
    const auto typeSize = static_cast<Byte>(getElemTypeSize(type));
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

//
// MemRefType utilities
//

namespace {

mlir::MemRefType addAffineMaps(mlir::MemRefType::Builder& memRefBuilder, mlir::MLIRContext* ctx, DimsOrder order,
                               ShapeRef shape) {
    // MLIR has canonicalization for default permutations such as affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    // in this case we will only have a strided map, which means the same as having no maps at all.
    // Let's skip strided map as it makes IR and test writing easier.
    if (order.isIdentity()) {
        return memRefBuilder.setAffineMaps({});
    }

    const auto affineMaps = order.toAffineMapsList(ctx, shape);
    return memRefBuilder.setAffineMaps(affineMaps);
}

}  // namespace

mlir::MemRefType vpux::changeElemType(mlir::MemRefType origType, mlir::Type elemType, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setElementType(elemType);

    mlir::MemRefType newType;
    if (preserveStrides) {
        newType = memRefBuilder;
    } else {
        const auto origOrder = DimsOrder::fromType(origType);
        const auto shape = getShape(origType);
        newType = addAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    }

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::changeShape(mlir::MemRefType origType, ShapeRef shape, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setShape(shape.raw());

    mlir::MemRefType newType;
    if (preserveStrides) {
        newType = memRefBuilder;
    } else {
        const auto origOrder = DimsOrder::fromType(origType);
        newType = addAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    }

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::changeDimsOrder(mlir::MemRefType origType, DimsOrder order) {
    mlir::MemRefType::Builder memRefBuilder(origType);

    const auto shape = getShape(origType);
    return addAffineMaps(memRefBuilder, origType.getContext(), order, shape);
}

mlir::MemRefType vpux::changeMemSpace(mlir::MemRefType origType, mlir::Attribute memSpace, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setMemorySpace(memSpace);

    if (preserveStrides) {
        return memRefBuilder;
    }

    const auto origOrder = DimsOrder::fromType(origType);
    const auto shape = getShape(origType);
    return addAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
}

mlir::MemRefType vpux::getDenseTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape) {
    return eraseTiledInfo(getViewTileType(origType, tileOffsets, tileShape));
}

mlir::MemRefType vpux::getViewTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape,
                                       ShapeRef tileElemStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);

    memRefBuilder.setShape(tileShape.raw());

    if (const auto perAxisQType = origType.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        memRefBuilder.setElementType(tileScalesAndZP(perAxisQType, tileShape, tileOffsets));
    }

    auto order = DimsOrder::fromType(origType);
    const auto elemSize = getElemTypeSize(origType);
    auto origStrides = getStrides(origType);

    auto memStrides = order.toMemoryOrder(origStrides);

    if (!tileElemStrides.empty()) {
        const auto memStridesMult = order.toMemoryOrder(tileElemStrides);

        VPUX_THROW_UNLESS(memStridesMult.size() == memStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides, memStrides.size());

        for (auto memInd1D : irange(memStrides.size())) {
            memStrides[MemDim(memInd1D)] *= memStridesMult[MemDim(memInd1D)];
        }
    }

    const auto affineMaps = order.toAffineMapsList(origType.getContext(), elemSize, memStrides);
    memRefBuilder.setAffineMaps(affineMaps);

    const mlir::MemRefType newType = memRefBuilder;

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::getPaddedType(mlir::MemRefType origType, ShapeRef padBefore, ShapeRef padAfter) {
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

    mlir::MemRefType::Builder memRefBuilder(origType);

    memRefBuilder.setShape(newShape.raw());

    if (const auto perAxisQType = origType.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        memRefBuilder.setElementType(expandScalesAndZP(perAxisQType, padBefore, padAfter));
    }

    const auto origOrder = DimsOrder::fromType(origType);
    const auto newType = addAffineMaps(memRefBuilder, origType.getContext(), origOrder, newShape);

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::eraseTiledInfo(mlir::MemRefType origType) {
    // Erase strides and offsets information from memory reference.
    const auto origShape = getShape(origType);
    const auto origOrder = DimsOrder::fromType(origType);

    mlir::MemRefType::Builder memRefBuilder(origType);
    return addAffineMaps(memRefBuilder, origType.getContext(), origOrder, origShape);
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ArrayRef<int64_t> shape, mlir::Type elementType, DimsOrder order,
                                           mlir::Attribute memSpace) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);

    const auto tensorDesc = IE::getTensorAttr(elementType.getContext(), order, memSpace);
    const auto newType = mlir::RankedTensorType::get(shape, elementType, tensorDesc);

    const auto loc = mlir::UnknownLoc::get(elementType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeElemType(mlir::RankedTensorType origType, mlir::Type elemType) {
    const auto newType =
            getTensorType(origType.getShape(), elemType, DimsOrder::fromType(origType), IE::getMemorySpace(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeShape(mlir::RankedTensorType origType, ShapeRef shape) {
    const auto origOrder = DimsOrder::fromType(origType);

    const auto newType = getTensorType(shape.raw(), origType.getElementType(),
                                       origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder,
                                       IE::getMemorySpace(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeDimsOrder(mlir::RankedTensorType origType, DimsOrder order) {
    return getTensorType(origType.getShape(), origType.getElementType(), order, IE::getMemorySpace(origType));
}

mlir::RankedTensorType vpux::changeMemSpace(mlir::RankedTensorType origType, mlir::Attribute memSpace) {
    return getTensorType(origType.getShape(), origType.getElementType(), DimsOrder::fromType(origType), memSpace);
}

mlir::RankedTensorType vpux::getDenseTileType(mlir::RankedTensorType origType, ShapeRef tileOffsets,
                                              ShapeRef tileShape) {
    auto elemType = origType.getElementType();
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto newType =
            getTensorType(tileShape.raw(), elemType, DimsOrder::fromType(origType), IE::getMemorySpace(origType));

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

    const auto newType =
            getTensorType(newShape.raw(), elemType, DimsOrder::fromType(origType), IE::getMemorySpace(origType));

    const auto loc = mlir::UnknownLoc::get(origType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
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

mlir::ShapedType vpux::changeMemSpace(mlir::ShapedType origType, mlir::Attribute memSpace) {
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
