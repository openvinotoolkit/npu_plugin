//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUDialect::registerTypes
//

void VPU::VPUDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU/generated/types.cpp.inc>
            >();
}

//
// VPU::DistributedTensorType accessors
//

ShapeRef VPU::DistributedTensorType::getShape() const {
    return ShapeRef(getImpl()->shape);
}

mlir::Type VPU::DistributedTensorType::getElementType() const {
    return getImpl()->elementType;
}

mlir::AffineMapAttr VPU::DistributedTensorType::getOrder() const {
    return getImpl()->order;
}

IndexedSymbolAttr VPU::DistributedTensorType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributedTensorAttr VPU::DistributedTensorType::getDistribution() const {
    return getImpl()->distribution;
}
