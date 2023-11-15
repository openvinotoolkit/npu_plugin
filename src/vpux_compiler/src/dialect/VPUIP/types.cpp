//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUIPDialect::registerTypes
//

void vpux::VPUIP::VPUIPDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
            >();
}

//
// BufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::BufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::BufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::BufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::BufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

mlir::IntegerAttr vpux::VPUIP::BufferType::getSwizzlingKey() const {
    return getImpl()->swizzlingKey;
}

//
// DistributedBufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::DistributedBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::DistributedBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::DistributedBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::DistributedBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributedTensorAttr vpux::VPUIP::DistributedBufferType::getDistribution() const {
    return getImpl()->distribution;
}

VPUIP::CompressionSchemeAttr vpux::VPUIP::DistributedBufferType::getCompressionScheme() const {
    return getImpl()->compressionScheme;
}
