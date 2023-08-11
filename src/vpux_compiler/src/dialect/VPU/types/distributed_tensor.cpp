//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
// DistributedTensorType::replaceImmediateSubElements
//

mlir::Type VPU::DistributedTensorType::replaceImmediateSubElements(ArrayRef<mlir::Attribute> replAttrs,
                                                                   ArrayRef<mlir::Type> replTypes) const {
    bool hasOrder = replAttrs.size() > 2;
    auto nextAfterOrder = hasOrder ? 1 : 0;
    size_t expectedSize = hasOrder ? 4 : 3;
    VPUX_THROW_WHEN(replAttrs.size() < expectedSize, "Replace attrs array is too short: '{0}'", replAttrs.size());
    return get(getContext(), getShape().raw(), replTypes[0],
               hasOrder ? replAttrs[0].dyn_cast_or_null<mlir::AffineMapAttr>() : mlir::AffineMapAttr(),
               replAttrs[nextAfterOrder].dyn_cast_or_null<vpux::IndexedSymbolAttr>(),
               replAttrs[nextAfterOrder + 1].dyn_cast_or_null<VPU::DistributedTensorAttr>());
}

//
// print/parse
//

void VPU::DistributedTensorType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();
    printer << ", " << getOrder();
    printer << ", " << getMemSpace();
    printer << ", {";

    auto distribution = getDistribution();
    printer << "mode = \"" << VPU::stringifyDistributionMode(distribution.mode().getValue()) << "\"";
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
    if (distribution.alignment() != nullptr) {
        printer << ", alignment = " << distribution.alignment();
    }
    if (distribution.uniform_distributed_segments() != nullptr) {
        printer << ", uniform_distributed_segments";
    }
    if (distribution.compute_shapes() != nullptr) {
        printer << ", compute_shapes = " << distribution.compute_shapes();
    }
    if (distribution.compute_offsets() != nullptr) {
        printer << ", compute_offsets = " << distribution.compute_offsets();
    }
    if (distribution.equal_memory_and_compute_view() != nullptr) {
        printer << ", equal_memory_and_compute_view";
    }
    printer << "}";

    printer << ">";
}

mlir::Type VPU::DistributedTensorType::parse(mlir::AsmParser& parser) {
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
    mlir::ArrayAttr alignment;
    mlir::UnitAttr uniformDistributedSegments;
    mlir::ArrayAttr computeShapes;
    mlir::ArrayAttr computeOffsets;
    mlir::UnitAttr equalComputeAndMemoryView;

    while (parser.parseOptionalRBrace()) {
        if (parser.parseComma()) {
            return Type();
        }
        std::string attrName;
        if (parser.parseKeywordOrString(&attrName)) {
            return Type();
        }

        // Handle UnitAttr first since they don't have value assigned
        if (attrName == "uniform_distributed_segments") {
            uniformDistributedSegments = mlir::UnitAttr::get(parser.getContext());
            continue;
        }

        if (attrName == "equal_memory_and_compute_view") {
            equalComputeAndMemoryView = mlir::UnitAttr::get(parser.getContext());
            continue;
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
        } else if (attrName == "alignment") {
            if (parser.parseAttribute(alignment)) {
                return Type();
            }
        } else if (attrName == "compute_shapes") {
            if (parser.parseAttribute(computeShapes)) {
                return Type();
            }
        } else if (attrName == "compute_offsets") {
            if (parser.parseAttribute(computeOffsets)) {
                return Type();
            }
        } else {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }
    auto distributedAttr = VPU::DistributedTensorAttr::get(
            distributionModeAttr, numTiles, kernel, pads, strides, numClusters, alignment, uniformDistributedSegments,
            computeShapes, computeOffsets, equalComputeAndMemoryView, parser.getContext());
    return static_cast<mlir::Type>(
            get(parser.getContext(), makeArrayRef(shape), elemType, order, memSpace, distributedAttr));
}

//
// verify
//

mlir::LogicalResult VPU::DistributedTensorType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       ::llvm::ArrayRef<int64_t> shape, mlir::Type /*elementType*/,
                                                       mlir::AffineMapAttr /*order*/, IndexedSymbolAttr /*memSpace*/,
                                                       DistributedTensorAttr distribution) {
    return VPU::verify(emitError, distribution, shape);
}

//
// getCompactType
//

mlir::RankedTensorType VPU::DistributedTensorType::getCompactType() const {
    return mlir::RankedTensorType::get(getShape().raw(), getElementType(),
                                       IE::TensorAttr::get(getOrder(), getMemSpace(), getContext()));
}

//
// Shape utils
//

// @brief Retrieve the array of compute shapes.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like SEGMENTED and OVERLAPPED take precedence over
//  DUPLICATED and MULTICASTED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
//  [1, 16, 4, 4], which is needed when tiling and generating workloads,
//  while the allocated shape is [1, 64, 4, 4] (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
//  In an example of OVERLAPPED over H with  k3x3s1x1 pad (1, 1, 1, 1) and
//  uniform segmentation for 4 clusters, a tensor of shape [1, 64, 22, 16]
//  will have the following compute distribution across clusters:
//  [1 64 6 16] [1 64 6 16] [1 64 5 16] [1 64 5 16]
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapes() const {
    return VPU::getPerClusterComputeShapes(getShape(), getDistribution());
}

// @brief Retrieve the offsets for each compute shape with regards to full tensor shape.
// @warning An important thing to consider with regards to compute offsets,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapeOffsets() const {
    return VPU::getPerClusterComputeShapeOffsets(getShape(), getDistribution());
}

// @brief Retrieve the array of memory shapes.
// @warning An important thing to consider with regards to memory shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the memory shape in each cluster is
//  [1, 64, 4, 4], which is the allocated shape (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
//  In an example of OVERLAPPED over H with k3x3s1x1 pad (1, 1, 1, 1) and
//  uniform segmentation across 4 clusters, a tensor of shape [1, 64, 22, 16]
//  will have the following memory distribution across clusters:
//  [1 64 7 16] [1 64 8 16] [1 64 7 16] [1 64 6 16]
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterMemoryShapes() const {
    return VPU::getPerClusterMemoryShapes(getShape(), getDistribution());
}

// @brief Retrieve the array of memory buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to memory shape offsets,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterMemoryShapeOffsets() const {
    return VPU::getPerClusterMemoryShapeOffsets(getShape(), getDistribution());
}

// @brief Get largest compact compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPU::DistributedTensorType::getLargestCompactShape() const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    return *std::max_element(tiledComputeShapes.begin(), tiledComputeShapes.end(), [](ShapeRef a, ShapeRef b) {
        return vpux::details::calcTotalShapeSize(a.raw()) < vpux::details::calcTotalShapeSize(b.raw());
    });
}

// @brief Get the compact compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPU::DistributedTensorType::getCompactShape(int64_t tileInd) const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(tiledComputeShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return tiledComputeShapes[tileInd];
}

// @brief Retrieve the array of padding for each cluster
// @warning This function is needed for getting padding in OVERLAPPED mode.
SmallVector<PadInfo> VPU::DistributedTensorType::getPerClusterPadding() const {
    return VPU::getPerClusterPadding(getDistribution());
}

// @brief Retrieve the array of strided memory shapes
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
SmallVector<StridedShape> VPU::DistributedTensorType::getPerClusterMemoryStridedShapes() const {
    const auto strideInReqs = StrideReqs::compact(getShape().size());
    VPUX_THROW_UNLESS(strideInReqs.checkStrides(*this), "Only compact strides are supported");
    return VPU::getPerClusterMemoryStridedShapes(getShape(), getStrides(), getDimsOrder(), getDistribution());
}

// @brief Get largest strided memory shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getLargestStridedShape() const {
    const auto stridedShapeSize = [](const StridedShape& stridedShape) {
        return stridedShape.shape.front() * stridedShape.strides.front();
    };

    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(!stridedShapes.empty(), "Missing per-cluster strided shapes");
    return *std::max_element(stridedShapes.begin(), stridedShapes.end(),
                             [&](const StridedShape& a, const StridedShape& b) {
                                 return stridedShapeSize(a) < stridedShapeSize(b);
                             });
}

// @brief Get the strided memory shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getStridedShape(int64_t tileInd) const {
    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(stridedShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return stridedShapes[tileInd];
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
    const auto perClusterShapes = getPerClusterMemoryShapes();
    const Shape tiledShape =
            *std::max_element(perClusterShapes.begin(), perClusterShapes.end(), [](ShapeRef a, ShapeRef b) {
                return vpux::details::calcTotalShapeSize(a.raw()) < vpux::details::calcTotalShapeSize(b.raw());
            });

    const auto totalSize = vpux::details::calcTotalShapeSize(tiledShape.raw());
    const auto elemSize = getElemTypeSize();
    const auto byteSize = static_cast<int64_t>(CHAR_BIT);
    if (elemSize.count() < byteSize) {
        return Byte(vpux::divUp(totalSize, byteSize));
    }

    return Byte(elemSize) * totalSize;
}

NDTypeInterface VPU::DistributedTensorType::changeShape(ShapeRef shape) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      getDimsOrder(), shape);
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

NDTypeInterface VPU::DistributedTensorType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      getDimsOrder(), shape);
    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
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

NDTypeInterface VPU::DistributedTensorType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("DistributedTensorType only supports compact strides");
}

NDTypeInterface VPU::DistributedTensorType::changeTypeComponents(TypeComponents typeComponents) const {
    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    auto elementType = getElementType();
    if (typeComponents.elementType.hasValue()) {
        elementType = typeComponents.elementType.getValue();
    } else {
        if (auto perAxisType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
            if (axis.hasValue()) {
                elementType = changeAxis(perAxisType, axis.getValue());
            }
        }
    }

    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elementType,
                                           mlir::AffineMapAttr::get(dimsOrder.toAffineMap(getContext())), memSpace,
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    auto elemType = getElementType();
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    return VPU::DistributedTensorType::get(getContext(), tileShape.raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::extractViewTile(vpux::ShapeRef /*tileOffsets*/,
                                                            vpux::ShapeRef /*tileShape*/,
                                                            vpux::ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("DistributedTensorType only supports compact strides");
}

NDTypeInterface VPU::DistributedTensorType::eraseTiledInfo() const {
    return *this;
}

NDTypeInterface VPU::DistributedTensorType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for DistributedTensorType");
}
