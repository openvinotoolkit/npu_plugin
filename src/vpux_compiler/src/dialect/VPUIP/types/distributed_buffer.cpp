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

void VPUIP::DistributedBufferType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                            llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
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

void VPUIP::DistributedBufferType::print(mlir::DialectAsmPrinter& printer) const {
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

mlir::Type VPUIP::DistributedBufferType::parse(mlir::DialectAsmParser& parser) {
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
// verify
//
mlir::LogicalResult VPUIP::DistributedBufferType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         ::llvm::ArrayRef<int64_t> /*shape*/,
                                                         mlir::Type /*elementType*/,
                                                         mlir::MemRefLayoutAttrInterface /*layout*/,
                                                         IndexedSymbolAttr /*memSpace*/,
                                                         VPU::DistributedTensorAttr distribution) {
    return VPU::verify(emitError, distribution);
}

//
// getCompactType
//

mlir::MemRefType VPUIP::DistributedBufferType::getCompactType() const {
    return mlir::MemRefType::get(getShape().raw(), getElementType(), getLayout(), getMemSpace());
}

//
// Shape utils
//

// @brief Retrive the array of compute shapes.
// @warning An important thing to consider with regards to compute shapes,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
// In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
// tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
// [1, 16, 4, 4], which is needed when tiling and generating workloads,
// while the allocated shape is [1, 64, 4, 4] (because of duplicated)
// information which is needed for scheduler and strategy manager,
// in order to estimate memory
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterComputeShapes() const {
    const auto shape = getShape();
    const auto strideInReqs = StrideReqs::compact(shape.size());
    VPUX_THROW_UNLESS(strideInReqs.checkStrides(*this), "Only compact strides are supported");

    return VPU::getPerClusterComputeShapes(getShape(), getDistribution());
}

// @brief Retrive the array of compute buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to compute shapes,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterComputeShapeOffsets() const {
    return VPU::getPerClusterComputeShapeOffsets(getShape(), getDistribution());
}

// @brief Get largest compact compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getLargestCompactShape() const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    return *std::max_element(tiledComputeShapes.begin(), tiledComputeShapes.end(), [](ShapeRef a, ShapeRef b) {
        return details::calcTotalShapeSize(a.raw()) < details::calcTotalShapeSize(b.raw());
    });
}

// @brief Get the compact compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getCompactShape(int64_t tileInd) const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(tiledComputeShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return tiledComputeShapes[tileInd];
}

// @brief Get largest strided compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getLargestStridedShape() const {
    VPUX_THROW("getLargestStridedShape method is not implemented for DistributedBufferType");
}

// @brief Get the strided compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getStridedShape(int64_t /*tileInd*/) const {
    VPUX_THROW("getStridedShape method is not implemented for DistributedBufferType");
}

//
// NDTypeInterface
//

MemShape VPUIP::DistributedBufferType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool VPUIP::DistributedBufferType::hasRank() const {
    return true;
}

int64_t VPUIP::DistributedBufferType::getRank() const {
    return checked_cast<int64_t>(getShape().size());
}

int64_t VPUIP::DistributedBufferType::getNumElements() const {
    auto shape = getShape().raw();
    VPUX_THROW_UNLESS(!details::isDynamicDimValues(shape), "Cannot get element count of dynamic shaped type");
    return details::calcTotalShapeSize(shape);
}

DimsOrder VPUIP::DistributedBufferType::getDimsOrder() const {
    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = layout.dyn_cast<IERT::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

VPU::MemoryKind VPUIP::DistributedBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }

    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

Strides VPUIP::DistributedBufferType::getStrides() const {
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

MemStrides VPUIP::DistributedBufferType::getMemStrides() const {
    const auto order = getDimsOrder();
    const auto strides = getStrides();
    return order.toMemoryOrder(strides);
}

Bit VPUIP::DistributedBufferType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPUIP::DistributedBufferType::getTotalAllocSize() const {
    return getCompactAllocSize();
}

Byte VPUIP::DistributedBufferType::getCompactAllocSize() const {
    auto shape = getShape();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.mode();

    // DUPLICATED|MULTICASTED takes priority since it means that each cluster will have the entire
    // tensor, regardless wheter it's tiled or not.
    ShapeRef tiledShape;
    if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::MULTICASTED)) {
        tiledShape = shape;
    } else if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::OVERLAPPED)) {
        tiledShape = getLargestCompactShape();
    } else {
        // No distribution mode.
        tiledShape = shape;
    }

    return Byte(getElemTypeSize()) * details::calcTotalShapeSize(tiledShape.raw());
}

NDTypeInterface VPUIP::DistributedBufferType::changeShape(ShapeRef) const {
    VPUX_THROW("changeShape method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeElemType(mlir::Type) const {
    VPUX_THROW("changeElemType method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeShapeElemType(ShapeRef, mlir::Type) const {
    VPUX_THROW("changeShapeElemType method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeDimsOrder(DimsOrder) const {
    VPUX_THROW("changeDimsOrder method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeMemSpace(IndexedSymbolAttr) const {
    VPUX_THROW("changeMemSpace method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::extractDenseTile(ShapeRef, ShapeRef) const {
    VPUX_THROW("extractDenseTile method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::pad(ShapeRef, ShapeRef) const {
    VPUX_THROW("pad method is not implemented for DistributedBufferType");
}
