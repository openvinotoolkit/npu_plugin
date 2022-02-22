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
    if (!getLayout().isIdentity()) {
        walkAttrsFn(getLayout());
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

    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        printer << ", " << mapAttr;
    } else if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        printer << ", " << descAttr;
    } else {
        VPUX_THROW("Unsupported MemRefType layout '{0}'", layout);
    }

    printer << ", " << getMemSpace();
    printer << ", {";

    auto distribution = getDistribution();
    printer << "mode = " << VPU::stringifyDistributionMode(distribution.mode().getValue());
    if (distribution.num_tiles() != nullptr) {
        printer << ", num_tiles = " << distribution.num_tiles();
    }
    if (distribution.kernel() != nullptr) {
        printer << ", kernel = " << distribution.kernel();
    }
    if (distribution.pads() != nullptr) {
        printer << ", pads = " << distribution.pads();
    }
    if (distribution.strides() != nullptr) {
        printer << ", strides = " << distribution.strides();
    }
    if (distribution.num_clusters() != nullptr) {
        printer << ", num_clusters = " << distribution.num_clusters();
    }
    printer << "}";

    printer << ">";
}

mlir::Type vpux::VPUIP::DistributedBufferType::parse(mlir::DialectAsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).hasValue() && parser.parseXInDimensionList().succeeded()) {
        shape.push_back(dim);
    }

    mlir::Type elemType;
    if (parser.parseType(elemType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    mlir::MemRefLayoutAttrInterface layout;

    mlir::AffineMapAttr mapAttr;
    IERT::MemRefAttr memRefAttr;
    if (parser.parseOptionalAttribute(mapAttr).hasValue()) {
        layout = mapAttr;
    } else if (parser.parseOptionalAttribute(memRefAttr).hasValue()) {
        layout = memRefAttr;
    } else {
        return Type();
    }

    if (parser.parseComma()) {
        return Type();
    }

    IndexedSymbolAttr memSpace;
    if (parser.parseAttribute(memSpace)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    // DistributedTensorAttr

    if (parser.parseLBrace()) {
        return Type();
    }

    // DistributionModeAttr

    if (parser.parseKeyword("mode")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    std::string distributionModeStr;
    if (parser.parseKeywordOrString(&distributionModeStr)) {
        return Type();
    }
    const auto distributionMode = VPU::symbolizeDistributionMode(distributionModeStr);
    if (!distributionMode.hasValue()) {
        return Type();
    }
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(parser.getContext(), distributionMode.getValue());

    mlir::ArrayAttr numTiles;
    mlir::ArrayAttr kernel;
    VPU::PaddingAttr pads;
    mlir::ArrayAttr strides;
    mlir::IntegerAttr numClusters;

    while (parser.parseOptionalRBrace()) {
        if (parser.parseComma()) {
            return Type();
        }
        std::string attrName;
        if (parser.parseKeywordOrString(&attrName)) {
            return Type();
        }
        if (parser.parseEqual()) {
            return Type();
        }
        if (attrName == "num_tiles") {
            if (parser.parseAttribute(numTiles)) {
                return Type();
            }
        } else if (attrName == "kernel") {
            if (parser.parseAttribute(kernel)) {
                return Type();
            }
        } else if (attrName == "pads") {
            if (parser.parseAttribute(pads)) {
                return Type();
            }
        } else if (attrName == "strides") {
            if (parser.parseAttribute(strides)) {
                return Type();
            }
        } else if (attrName == "num_clusters") {
            if (parser.parseAttribute(numClusters)) {
                return Type();
            }
        } else {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }
    auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTiles, kernel, pads, strides,
                                                           numClusters, parser.getContext());
    return static_cast<mlir::Type>(
            get(parser.getContext(), makeArrayRef(shape), elemType, layout, memSpace, distributedAttr));
}

//
// getCompactType
//

mlir::MemRefType vpux::VPUIP::DistributedBufferType::getCompactType() const {
    return mlir::MemRefType::get(getShape().raw(), getElementType(), getLayout(), getMemSpace());
}

//
// NDTypeInterface
//

vpux::MemShape vpux::VPUIP::DistributedBufferType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
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
    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

vpux::VPU::MemoryKind vpux::VPUIP::DistributedBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides vpux::VPUIP::DistributedBufferType::getStrides() const {
    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);

        // Missing strides specification means compact strides.
        const auto order = getDimsOrder();
        const auto memShape = getMemShape();
        const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

        return order.toLogicalOrder(memStrides);
    }

    if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        const auto elemStrides = parseIntArrayAttr<int64_t>(descAttr.strides());
        const auto elemSize = getElemTypeSize();

        return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                           return stride * elemSize;
                                       })));
    }

    VPUX_THROW("Unsupported layout attribute type '{0}'", layout);
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
    VPUX_THROW("changeShape method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeElemType(mlir::Type) const {
    VPUX_THROW("changeElemType method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeDimsOrder(vpux::DimsOrder) const {
    VPUX_THROW("changeDimsOrder method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::changeMemSpace(vpux::IndexedSymbolAttr) const {
    VPUX_THROW("changeMemSpace method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::extractDenseTile(vpux::ShapeRef, vpux::ShapeRef) const {
    VPUX_THROW("extractDenseTile method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPUIP::DistributedBufferType::pad(vpux::ShapeRef, vpux::ShapeRef) const {
    VPUX_THROW("pad method is not implemented for DistributedBufferType");
}
