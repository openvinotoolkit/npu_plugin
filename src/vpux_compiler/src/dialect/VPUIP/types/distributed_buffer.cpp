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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// SubElementTypeInterface
//

void vpux::VPUIP::DistributedBufferType::walkImmediateSubElements(
        llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    walkTypesFn(getElementType());
    if (!getOrder().isIdentity()) {
        walkAttrsFn(getOrder());
    }
    walkAttrsFn(getMemSpace());
    walkAttrsFn(getDistribution());
}

//
// print/parse
//

void vpux::VPUIP::DistributedBufferType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();
    printer << ", " << getOrder();
    printer << ", " << getMemSpace();
    printer << ", " << getDistribution();
    printer << ">";
}

mlir::Type vpux::VPUIP::DistributedBufferType::parse(mlir::DialectAsmParser& parser) {
    if (parser.parseLess()) {
        return mlir::Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).hasValue() && parser.parseXInDimensionList().succeeded()) {
        shape.push_back(dim);
    }

    mlir::Type elemType;
    if (parser.parseType(elemType)) {
        return mlir::Type();
    }
    if (parser.parseComma()) {
        return mlir::Type();
    }

    mlir::AffineMapAttr order;
    if (parser.parseAttribute(order)) {
        return mlir::Type();
    }
    if (parser.parseComma()) {
        return mlir::Type();
    }

    vpux::IndexedSymbolAttr memSpace;
    if (parser.parseAttribute(memSpace)) {
        return mlir::Type();
    }
    if (parser.parseComma()) {
        return mlir::Type();
    }

    VPU::DistributedTensorAttr distribution;
    if (parser.parseAttribute(distribution)) {
        return mlir::Type();
    }
    if (parser.parseGreater()) {
        return mlir::Type();
    }

    return get(parser.getContext(), makeArrayRef(shape), elemType, order, memSpace, distribution);
}

//
// getCompactType
//

mlir::MemRefType vpux::VPUIP::DistributedBufferType::getCompactType() const {
    return mlir::MemRefType::get(getShape().raw(), getElementType(), getOrder().getValue(), getMemSpace());
}

//
// NDTypeInterface
//

vpux::MemShape vpux::VPUIP::DistributedBufferType::getMemShape() const {
    const auto dimsOrder = DimsOrder::fromAffineMap(getOrder().getValue());
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool vpux::VPUIP::DistributedBufferType::hasRank() const {
    return true;
}

int64_t vpux::VPUIP::DistributedBufferType::getRank() const {
    return checked_cast<int64_t>(getShape().size());
}

int64_t vpux::VPUIP::DistributedBufferType::getNumElements() const {
    auto shape = getShape().raw();
    VPUX_THROW_UNLESS(!vpux::details::isDynamicDimValues(shape), "Cannot get element count of dynamic shaped type");
    return vpux::details::calcTotalShapeSize(shape);
}

vpux::DimsOrder vpux::VPUIP::DistributedBufferType::getDimsOrder() const {
    return DimsOrder::fromAffineMap(getOrder().getValue());
}

vpux::VPU::MemoryKind vpux::VPUIP::DistributedBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides vpux::VPUIP::DistributedBufferType::getStrides() const {
    const auto mapAttr = getOrder();
    VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", mapAttr);

    const auto order = getDimsOrder();
    const auto memShape = getMemShape();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides vpux::VPUIP::DistributedBufferType::getMemStrides() const {
    const auto order = getDimsOrder();
    const auto strides = getStrides();
    return order.toMemoryOrder(strides);
}

vpux::Bit vpux::VPUIP::DistributedBufferType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

vpux::Byte vpux::VPUIP::DistributedBufferType::getTotalAllocSize() const {
    if (getRank() == 0) {
        return getElemTypeSize();
    }

    const auto memShape = getMemShape();
    const auto memStrides = getMemStrides();

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

vpux::Byte vpux::VPUIP::DistributedBufferType::getCompactAllocSize() const {
    const auto typeSize = static_cast<Bit>(getElemTypeSize());
    if (getRank() == 0) {
        return typeSize;
    }

    const auto shape = getShape();
    return shape.totalSize() * typeSize;
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeShape(vpux::ShapeRef) const {
    VPUX_THROW("Not yet implemented");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeElemType(mlir::Type) const {
    VPUX_THROW("Not yet implemented");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeDimsOrder(vpux::DimsOrder) const {
    VPUX_THROW("Not yet implemented");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeMemSpace(vpux::IndexedSymbolAttr) const {
    VPUX_THROW("Not yet implemented");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::extractDenseTile(vpux::ShapeRef, vpux::ShapeRef) const {
    VPUX_THROW("Not yet implemented");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::pad(vpux::ShapeRef, vpux::ShapeRef) const {
    VPUX_THROW("Not yet implemented");
}
