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

Byte vpux::getTypeTotalSize(mlir::MemRefType type) {
    if (type.getRank() == 0) {
        return getElemTypeSize(type);
    }

    const auto dimsOrder = DimsOrder::fromType(type);
    const auto shape = getShape(type);
    const auto strides = getStrides(type);
    const auto memShape = dimsOrder.toMemoryOrder(shape);
    const auto memStrides = dimsOrder.toMemoryOrder(strides);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Size and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast_or_null<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non MemRefType '{1}'", val, val.getType());
    return getTypeTotalSize(type);
}

//
// MemRefType utilities
//

namespace {

void setAffineMaps(mlir::MemRefType::Builder& memRefBuilder, mlir::MLIRContext* ctx, DimsOrder order, ShapeRef shape) {
    // MLIR has canonicalization for default permutations such as affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    // in this case we will only have a strided map, which means the same as having no maps at all.
    // Let's skip strided map as it makes IR and test writing easier.
    if (order.isIdentity()) {
        memRefBuilder.setAffineMaps({});
        return;
    }

    const auto affineMaps = order.toAffineMapsList(ctx, shape);
    memRefBuilder.setAffineMaps(affineMaps);
}

}  // namespace

mlir::MemRefType vpux::changeElemType(mlir::MemRefType origType, mlir::Type elemType, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setElementType(elemType);

    if (!preserveStrides) {
        const auto origOrder = DimsOrder::fromType(origType);
        const auto shape = getShape(origType);
        setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    }

    return memRefBuilder;
}

mlir::MemRefType vpux::changeShape(mlir::MemRefType origType, ShapeRef shape, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setShape(shape.raw());

    if (!preserveStrides) {
        const auto origOrder = DimsOrder::fromType(origType);
        setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    }

    return memRefBuilder;
}

mlir::MemRefType vpux::changeDimsOrder(mlir::MemRefType origType, DimsOrder order) {
    mlir::MemRefType::Builder memRefBuilder(origType);

    const auto shape = getShape(origType);
    setAffineMaps(memRefBuilder, origType.getContext(), order, shape);

    return memRefBuilder;
}

mlir::MemRefType vpux::changeMemSpace(mlir::MemRefType origType, mlir::Attribute memSpace, bool preserveStrides) {
    mlir::MemRefType::Builder memRefBuilder(origType);
    memRefBuilder.setMemorySpace(memSpace);

    if (!preserveStrides) {
        const auto origOrder = DimsOrder::fromType(origType);
        const auto shape = getShape(origType);
        setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    }

    return memRefBuilder;
}

mlir::MemRefType vpux::getTileType(const mlir::MemRefType origType, const ShapeRef tileShape,
                                   const ShapeRef tileOffsets) {
    const auto strides = getStrides(origType);
    Bit totalOffset(0);
    for (size_t dimIdx = 0; dimIdx < strides.size(); dimIdx++) {
        totalOffset += tileOffsets[Dim(dimIdx)] * strides[Dim(dimIdx)];
    }
    const Bit elemSize = getElemTypeSize(origType);
    const auto affineMaps = DimsOrder::fromType(origType).toAffineMapsList(origType.getContext(), getShape(origType),
                                                                           totalOffset.count() / elemSize.count());

    const auto tileType =
            mlir::MemRefType::get(tileShape.raw(), origType.getElementType(), affineMaps, origType.getMemorySpace());
    return tileType;
}

mlir::MemRefType vpux::eraseTiledInfo(const mlir::MemRefType origType) {
    // Erase strides and offsets information from memory reference.
    const auto origShape = getShape(origType);
    const auto origOrder = DimsOrder::fromType(origType);
    mlir::MemRefType::Builder memRefBuilder(origType);
    setAffineMaps(memRefBuilder, origType.getContext(), origOrder, origShape);

    return memRefBuilder;
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ArrayRef<int64_t> shape, mlir::Type elementType, DimsOrder order) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);
    return mlir::RankedTensorType::get(shape, elementType, IE::getTensorAttr(elementType.getContext(), order));
}

mlir::RankedTensorType vpux::changeElemType(mlir::RankedTensorType origType, mlir::Type elemType) {
    return getTensorType(origType.getShape(), elemType, DimsOrder::fromType(origType));
}

mlir::RankedTensorType vpux::changeShape(mlir::RankedTensorType origType, ShapeRef shape) {
    const auto origOrder = DimsOrder::fromType(origType);
    return getTensorType(shape.raw(), origType.getElementType(),
                         origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder);
}

mlir::RankedTensorType vpux::changeDimsOrder(mlir::RankedTensorType origType, DimsOrder order) {
    return getTensorType(origType.getShape(), origType.getElementType(), order);
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
