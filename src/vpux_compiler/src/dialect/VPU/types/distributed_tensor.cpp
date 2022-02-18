//
// Copyright 2022 Intel Corporation
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

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// SubElementTypeInterface
//

void VPU::DistributedTensorType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(Type)> walkTypesFn) const {
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

void VPU::DistributedTensorType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();
    printer << ", " << getOrder();
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

mlir::Type VPU::DistributedTensorType::parse(mlir::DialectAsmParser& parser) {
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

    mlir::AffineMapAttr order;
    if (parser.parseAttribute(order)) {
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
            get(parser.getContext(), makeArrayRef(shape), elemType, order, memSpace, distributedAttr));
}

//
// getCompactType
//

mlir::RankedTensorType vpux::VPU::DistributedTensorType::getCompactType() const {
    return mlir::RankedTensorType::get(getShape().raw(), getElementType(),
                                       IE::TensorAttr::get(getOrder(), getMemSpace(), nullptr, getContext()));
}

//
// NDTypeInterface
//

MemShape VPU::DistributedTensorType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool VPU::DistributedTensorType::hasRank() const {
    return true;
}

int64_t VPU::DistributedTensorType::getRank() const {
    return static_cast<int64_t>(getShape().size());
}

DimsOrder VPU::DistributedTensorType::getDimsOrder() const {
    return DimsOrder::fromAffineMap(getOrder().getValue());
}

int64_t VPU::DistributedTensorType::getNumElements() const {
    return vpux::details::calcTotalShapeSize(getShape().raw());
}

VPU::MemoryKind VPU::DistributedTensorType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }
    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

Strides VPU::DistributedTensorType::getStrides() const {
    const auto mapAttr = getOrder();
    VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", mapAttr);

    const auto memStrides = getMemStrides();
    const auto order = getDimsOrder();
    return order.toLogicalOrder(memStrides);
}

MemStrides VPU::DistributedTensorType::getMemStrides() const {
    const auto order = getDimsOrder();
    // Tensors are always compact
    const auto elemSize = getElemTypeSize();
    const auto shape = getShape();
    const auto memShape = order.toMemoryOrder(shape);
    return StrideReqs::compact(order.numDims()).calcStrides(elemSize, memShape);
}

Bit VPU::DistributedTensorType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPU::DistributedTensorType::getTotalAllocSize() const {
    // Tensors are always compact
    return getCompactAllocSize();
}

Byte VPU::DistributedTensorType::getCompactAllocSize() const {
    auto shape = getShape();
    const auto distribution = getDistribution();
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.num_tiles());
    const auto distributionMode = distribution.mode();

    auto tiledShape = SmallVector<int64_t>(shape.size());
    if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::SEGMENTED)) {
        std::transform(shape.begin(), shape.end(), tilingScheme.begin(), tiledShape.begin(),
                       [](int64_t dim, int64_t tile) {
                           return divUp(dim, tile);
                       });
    } else if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::OVERLAPPED)) {
        VPUX_THROW("OVERLAPPED distribution mode is not supported yet");
    } else {
        tiledShape = to_small_vector(shape);
    }

    return Byte(getElemTypeSize()) * vpux::details::calcTotalShapeSize(tiledShape);
}

NDTypeInterface VPU::DistributedTensorType::changeShape(ShapeRef shape) const {
    auto elemType = getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
        if (axis.hasValue()) {
            elemType = changeAxis(perAxisType, axis.getValue());
        }
    }
    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeElemType(mlir::Type elemType) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeDimsOrder(DimsOrder order) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), getElementType(),
                                           mlir::AffineMapAttr::get(order.toAffineMap(getContext())), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), getElementType(), getOrder(), memSpace,
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::extractDenseTile(ShapeRef /*tileOffsets*/, ShapeRef /*tileShape*/) const {
    VPUX_THROW("extractDenseTile method is not implemented for DistributedTensorType");
}

NDTypeInterface VPU::DistributedTensorType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for DistributedTensorType");
}
