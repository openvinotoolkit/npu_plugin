//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <numeric>

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/attributes.cpp.inc>

#include <vpux/compiler/dialect/VPUIP/enums.cpp.inc>

using namespace vpux;

//
// Dialect hooks
//

void VPUIP::VPUIPDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPUIP/attributes.cpp.inc>
            >();
}

//
// CompressionSchemeAttr
//

int64_t VPUIP::CompressionSchemeAttr::getTotalNumElems() const {
    if (getNumElems().empty()) {
        return 0;
    }
    auto numElems = getNumElems().getValues<int64_t>();
    return std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0));
}

int64_t VPUIP::CompressionSchemeAttr::getNumElemsInRange(int64_t startIdx, int64_t size) const {
    const auto numElems = getNumElems().getValues<int64_t>();
    const auto startIt = numElems.begin() + startIdx;
    const auto endIt = startIt + size;
    return std::accumulate(startIt, endIt, static_cast<int64_t>(0));
}

Byte VPUIP::CompressionSchemeAttr::getAllocSize(mlir::Type elemType) const {
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const int64_t alignment = (getAlignment() != nullptr) ? getAlignment().getInt() : 1;
    const auto numElems = getNumElems().getValues<int64_t>();
    int64_t totalAllocSize = 0;
    for (auto num : numElems) {
        totalAllocSize += alignValUp<int64_t>(num * elemByteSize, alignment);
    }
    return Byte(totalAllocSize);
}

VPUIP::CompressionSchemeAttr VPUIP::getCompressionSchemeAttr(mlir::Type type) {
    if (type == nullptr) {
        return nullptr;
    }

    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
        if (const auto memRefAttr = memref.getLayout().dyn_cast_or_null<vpux::MemRefAttr>()) {
            return memRefAttr.hwSpecificField<VPUIP::CompressionSchemeAttr>();
        }
    } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        return distributedBuffer.getCompressionScheme();
    } else if (auto sparseType = type.dyn_cast<VPUIP::SparseBufferType>()) {
        return sparseType.getCompressionScheme();
    }

    return nullptr;
}

mlir::Type VPUIP::setCompressionSchemeAttr(mlir::Type type, VPUIP::CompressionSchemeAttr compressionSchemeAttr) {
    if (type == nullptr) {
        return nullptr;
    }

    if (type.isa<mlir::MemRefType>()) {
        auto ndType = type.cast<vpux::NDTypeInterface>();
        return getMemRefType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(), ndType.getMemSpace(),
                             ndType.getStrides(), getSwizzlingSchemeAttr(type), compressionSchemeAttr);
    } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        return VPUIP::DistributedBufferType::get(type.getContext(), distributedBuffer.getShape().raw(),
                                                 distributedBuffer.getElementType(), distributedBuffer.getLayout(),
                                                 distributedBuffer.getMemSpace(), distributedBuffer.getDistribution(),
                                                 compressionSchemeAttr);
    } else if (auto sparseType = type.dyn_cast<VPUIP::SparseBufferType>()) {
        return VPUIP::SparseBufferType::get(sparseType.getData(), sparseType.getSparsityMap(),
                                            sparseType.getStorageElementTable(), sparseType.getIsWeights(),
                                            compressionSchemeAttr);
    }

    return type;
}

VPUIP::CompressionSchemeAttr VPUIP::tileCompressionScheme(VPUIP::CompressionSchemeAttr compressionScheme,
                                                          ShapeRef tileOffsets, ShapeRef tileShape) {
    if (compressionScheme == nullptr) {
        return nullptr;
    }
    VPUX_THROW_UNLESS(compressionScheme.getAxis() != nullptr,
                      "Cannot tile compression scheme that is not over an axis");
    const size_t axis = compressionScheme.getAxis().getInt();
    VPUX_THROW_UNLESS(axis < tileOffsets.size() && axis < tileShape.size(),
                      "Axis {0} outside the range of tile dimensions: offsets size {1}, shape size {2}", axis,
                      tileOffsets.size(), tileShape.size());

    const auto numElems = compressionScheme.getNumElems().getValues<int64_t>();
    const auto dimOffset = tileOffsets[Dim(axis)];
    const auto dimShape = tileShape[Dim(axis)];

    const auto startIt = numElems.begin() + dimOffset;
    const auto endIt = startIt + dimShape;
    const auto tileNumElems = SmallVector<int64_t>(startIt, endIt);

    auto ctx = compressionScheme.getContext();
    const auto tileNumElemsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(tileNumElems.size())}, getInt64Type(ctx));
    const auto tileNumElemsAttr = mlir::DenseElementsAttr::get(tileNumElemsType, ArrayRef(tileNumElems));
    return VPUIP::CompressionSchemeAttr::get(ctx, compressionScheme.getAxis(), tileNumElemsAttr,
                                             compressionScheme.getAlignment());
}

mlir::Type VPUIP::tileTypeCompressionScheme(mlir::Type type, ShapeRef tileOffsets, ShapeRef tileShape) {
    if (type == nullptr) {
        return nullptr;
    }

    const auto compressionScheme = VPUIP::getCompressionSchemeAttr(type);
    if (compressionScheme == nullptr) {
        return type;
    }

    const auto tiledCompressionScheme = VPUIP::tileCompressionScheme(compressionScheme, tileOffsets, tileShape);
    return VPUIP::setCompressionSchemeAttr(type, tiledCompressionScheme);
}
