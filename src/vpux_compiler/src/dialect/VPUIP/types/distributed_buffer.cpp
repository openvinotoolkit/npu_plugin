//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <numeric>

using namespace vpux;

namespace {
vpux::MemRefAttr::HwFields getHwSpecificFields(mlir::MemRefLayoutAttrInterface layout) {
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        return memRefAttr.hwSpecificFields();
    }
    return {};
}
}  // namespace

//
// print/parse
//

void VPUIP::DistributedBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();

    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        printer << ", " << mapAttr;
    } else if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        printer << ", " << descAttr;
    } else {
        VPUX_THROW("Unsupported MemRefType layout '{0}'", layout);
    }

    printer << ", " << getMemSpace();

    printer << ", {";

    auto distribution = getDistribution();
    printer << "mode = \"" << VPU::stringifyDistributionMode(distribution.getMode().getValue()) << "\"";
    if (distribution.getNumTiles() != nullptr) {
        printer << ", num_tiles = " << distribution.getNumTiles();
    }
    if (distribution.getKernel() != nullptr) {
        printer << ", kernel = " << distribution.getKernel();
    }
    if (distribution.getPads() != nullptr) {
        printer << ", pads = " << distribution.getPads();
    }
    if (distribution.getStrides() != nullptr) {
        printer << ", strides = " << distribution.getStrides();
    }
    if (distribution.getNumClusters() != nullptr) {
        printer << ", num_clusters = " << distribution.getNumClusters();
    }
    if (distribution.getAlignment() != nullptr) {
        printer << ", alignment = " << distribution.getAlignment();
    }
    if (distribution.getUniformDistributedSegments() != nullptr) {
        printer << ", uniform_distributed_segments";
    }
    if (distribution.getComputeShapes() != nullptr) {
        printer << ", compute_shapes = " << distribution.getComputeShapes();
    }
    if (distribution.getComputeOffsets() != nullptr) {
        printer << ", compute_offsets = " << distribution.getComputeOffsets();
    }
    if (distribution.getMemoryShapes() != nullptr) {
        printer << ", memory_shapes = " << distribution.getMemoryShapes();
    }
    if (distribution.getMemoryOffsets() != nullptr) {
        printer << ", memory_offsets = " << distribution.getMemoryOffsets();
    }
    if (distribution.getEqualMemoryAndComputeView() != nullptr) {
        printer << ", equal_memory_and_compute_view";
    }
    printer << "}";
    if (getCompressionScheme() != nullptr) {
        printer << ", " << getCompressionScheme();
    }

    printer << ">";
}

mlir::Type VPUIP::DistributedBufferType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).has_value() && parser.parseXInDimensionList().succeeded()) {
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
    vpux::MemRefAttr memRefAttr;
    if (parser.parseOptionalAttribute(mapAttr).has_value()) {
        layout = mapAttr;
    } else if (parser.parseOptionalAttribute(memRefAttr).has_value()) {
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
    if (!distributionMode.has_value()) {
        return Type();
    }
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(parser.getContext(), distributionMode.value());

    mlir::ArrayAttr numTiles;
    mlir::ArrayAttr kernel;
    VPU::PaddingAttr pads;
    mlir::ArrayAttr strides;
    mlir::IntegerAttr numClusters;
    mlir::ArrayAttr alignment;
    mlir::UnitAttr uniformDistributedSegments;
    mlir::ArrayAttr computeShapes;
    mlir::ArrayAttr computeOffsets;
    mlir::ArrayAttr memoryShapes;
    mlir::ArrayAttr memoryOffsets;
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
        } else if (attrName == "equal_memory_and_compute_view") {
            equalComputeAndMemoryView = mlir::UnitAttr::get(parser.getContext());
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
        } else if (attrName == "compute_offsets") {
            if (parser.parseAttribute(computeOffsets)) {
                return Type();
            }
        } else if (attrName == "compute_shapes") {
            if (parser.parseAttribute(computeShapes)) {
                return Type();
            }
        } else if (attrName == "memory_offsets") {
            if (parser.parseAttribute(memoryOffsets)) {
                return Type();
            }
        } else if (attrName == "memory_shapes") {
            if (parser.parseAttribute(memoryShapes)) {
                return Type();
            }
        } else {
            return Type();
        }
    }

    VPUIP::CompressionSchemeAttr compressionScheme;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (parser.parseAttribute(compressionScheme)) {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }
    auto distributedAttr =
            VPU::DistributedTensorAttr::get(parser.getContext(), distributionModeAttr, numTiles, kernel, pads, strides,
                                            numClusters, alignment, uniformDistributedSegments, computeShapes,
                                            computeOffsets, memoryShapes, memoryOffsets, equalComputeAndMemoryView);
    return static_cast<mlir::Type>(
            get(parser.getContext(), ArrayRef(shape), elemType, layout, memSpace, distributedAttr, compressionScheme));
}

//
// verify
//

mlir::LogicalResult VPUIP::DistributedBufferType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                                                         mlir::MemRefLayoutAttrInterface layout,
                                                         IndexedSymbolAttr /*memSpace*/,
                                                         VPU::DistributedTensorAttr distribution,
                                                         VPUIP::CompressionSchemeAttr compressionScheme) {
    if (mlir::failed(VPU::verify(emitError, distribution, shape))) {
        return mlir::failure();
    }

    if (compressionScheme != nullptr) {
        if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
            const Bit elemTypeSize = vpux::getElemTypeSize(elementType);
            if (auto stridesAttr = descAttr.strides()) {
                const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
                const auto strides = Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                                                 return stride * elemTypeSize;
                                                             })));
                const auto order = DimsOrder::fromAffineMap(descAttr.order().getValue());
                const auto memShape = order.toMemoryOrder(Shape(shape));
                const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(elemTypeSize, memShape);
                const auto compactStrides = order.toLogicalOrder(memStrides);
                if (strides != compactStrides) {
                    return printTo(emitError(), "Cannot compress strided buffer");
                }
            }
        }

        const auto distributionMode = distribution.getMode().getValue();
        if (distributionMode != VPU::DistributionMode::SEGMENTED &&
            distributionMode != VPU::DistributionMode::OVERLAPPED) {
            return mlir::success();
        }
        if (compressionScheme.getAxis() == nullptr) {
            return printTo(emitError(), "Cannot compressed the entire buffer for SEGMENTED/OVERLAPPED modes");
        }
        const auto axis = compressionScheme.getAxis().getInt();
        if (axis != Dims4D::Filter::OC.ind()) {
            return printTo(emitError(),
                           "Only constants can be compressed and the compression can only be done over OC");
        }
        auto tilesOnAxis = parseIntArrayAttr<int64_t>(distribution.getNumTiles())[axis];
        if (tilesOnAxis == 1) {
            return printTo(emitError(), "Cannot segment and compress buffer on different dimensions");
        }
    }

    return mlir::success();
}

//
// getCompactType
//

mlir::MemRefType VPUIP::DistributedBufferType::getCompactType() const {
    auto distributionMode = getDistribution().getMode().getValue();
    auto swizzlingSchemeAttr = vpux::getSwizzlingSchemeAttr(*this);

    // Compact shapes from SEGMENTED/OVERLAPPED buffers may require more memory after their individual sizeAlignment and
    // thus the producer segmented buffer may require extra adjustment in sizeAlignment
    if (swizzlingSchemeAttr != nullptr && (distributionMode == VPU::DistributionMode::SEGMENTED ||
                                           distributionMode == VPU::DistributionMode::OVERLAPPED)) {
        const auto distributionAttr = getDistribution();
        VPUX_THROW_UNLESS(distributionAttr.getNumClusters() != distributionAttr.getNumTiles(),
                          "Unsupported case to re-align compact buffer with swizzling");

        const auto ctx = getContext();
        swizzlingSchemeAttr =
                VPUIP::SwizzlingSchemeAttr::get(ctx, swizzlingSchemeAttr.getKey(),
                                                getIntAttr(ctx, swizzlingSchemeAttr.getSizeAlignment().getInt() *
                                                                        distributionAttr.getNumClusters().getInt()));
    }

    return vpux::getMemRefType(getShape(), getElementType(), getDimsOrder(), getMemSpace(), getStrides(),
                               swizzlingSchemeAttr, VPUIP::getCompressionSchemeAttr(*this));
}

//
// Shape utils
//

namespace {

Shape* getLargestShapeIt(SmallVector<Shape>& shapes) {
    return std::max_element(shapes.begin(), shapes.end(), [](ShapeRef a, ShapeRef b) {
        return details::calcTotalShapeSize(a.raw()) < details::calcTotalShapeSize(b.raw());
    });
}

StridedShape* getLargestStridedShapeIt(SmallVector<StridedShape>& stridedShapes) {
    const auto stridedShapeSize = [](const StridedShape& stridedShape) {
        return stridedShape.shape.front() * stridedShape.strides.front();
    };
    return std::max_element(stridedShapes.begin(), stridedShapes.end(),
                            [&](const StridedShape& a, const StridedShape& b) {
                                return stridedShapeSize(a) < stridedShapeSize(b);
                            });
}

}  // namespace

// @brief Retrieve the array of compute shapes.
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
    auto distribution = getDistribution();
    if (distribution.getComputeShapes() == nullptr) {
        return VPU::getPerClusterComputeShapes(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeShapes());
}

// @brief Retrieve the array of compute buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to compute shapes,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterComputeShapeOffsets() const {
    auto distribution = getDistribution();
    if (distribution.getComputeOffsets() == nullptr) {
        return VPU::getPerClusterComputeShapeOffsets(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeOffsets());
}

// @brief Retrieve the array of memory shapes.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the memory shape in each cluster is
//  [1, 64, 4, 4], which is the allocated shape (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterMemoryShapes() const {
    auto distribution = getDistribution();
    if (distribution.getMemoryShapes() == nullptr) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
        return optionalPerClusterMemoryShapes.value();
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getMemoryShapes());
}

// @brief Retrieve the array of memory buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterMemoryShapeOffsets() const {
    auto distribution = getDistribution();
    if (distribution.getMemoryOffsets() == nullptr) {
        return VPU::getPerClusterMemoryShapeOffsets(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getMemoryOffsets());
}

// @brief Get largest compact compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getLargestCompactShape() const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    return *getLargestShapeIt(tiledComputeShapes);
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

// @brief Retrieve the array of padding for each cluster
// @warning This function is needed for getting padding in OVERLAPPED mode.
SmallVector<PadInfo> VPUIP::DistributedBufferType::getPerClusterPadding(PadInfo kernelPadding) const {
    return VPU::getPerClusterPadding(getDistribution(), kernelPadding);
}

// @brief Retrieve the array of strided memory shapes
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
SmallVector<StridedShape> VPUIP::DistributedBufferType::getPerClusterMemoryStridedShapes() const {
    const auto memoryShapes = getPerClusterMemoryShapes();
    return VPU::getPerClusterMemoryStridedShapes(getShape(), getStrides(), getDimsOrder(), getDistribution().getMode(),
                                                 memoryShapes);
}

// @brief Get largest strided compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPUIP::DistributedBufferType::getLargestStridedShape() const {
    auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(!stridedShapes.empty(), "Missing per-cluster strided shapes");
    return *getLargestStridedShapeIt(stridedShapes);
}

// @brief Get the strided compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPUIP::DistributedBufferType::getStridedShape(int64_t tileInd) const {
    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(stridedShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return stridedShapes[tileInd];
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offets adjusted for the new shape.
NDTypeInterface VPUIP::DistributedBufferType::changeShapeForExplicitDistribution(
        ShapeRef shape, VPU::DistributedTensorAttr distributedAttr) const {
    return changeShapeElemTypeForExplicitDistribution(shape, getElementType(), distributedAttr);
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and element type and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offets adjusted for the new shape.
NDTypeInterface VPUIP::DistributedBufferType::changeShapeElemTypeForExplicitDistribution(
        ShapeRef shape, mlir::Type elemType, VPU::DistributedTensorAttr distributedAttr) const {
    const auto ctx = getContext();
    const auto origOrder = getDimsOrder();
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto layoutAttr = getLayout();
    if (auto memRefAttr = getLayout().dyn_cast<vpux::MemRefAttr>()) {
        const auto orderAttr = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
        // If swizzlingKey is set get rid of strides settings
        if (memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>()) {
            layoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr,
                                               /*allocSize=*/nullptr, memRefAttr.hwSpecificFields(), ctx);
        } else {
            layoutAttr = orderAttr;
        }
    }

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, getMemSpace(),
                                                     distributedAttr, getCompressionScheme());

    const auto loc = mlir::UnknownLoc::get(ctx);
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "ChangeShape caused mismatch with quantization settings'{0}'", newType);

    return newType;
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested type components. If shape is one of the changed
// components, it will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offets
// adjusted for the new shape. Otherwise, it leaves the DistributedAttr untouched.
NDTypeInterface VPUIP::DistributedBufferType::changeTypeComponentsForExplicitDistribution(
        const TypeComponents& typeComponents, VPU::DistributedTensorAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return changeTypeComponents(typeComponents);
    }

    const auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType());
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto strides = typeComponents.strides.value_or(getStrides());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    VPUX_THROW_UNLESS(dimsOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the shape '{1}'",
                      dimsOrder, shape);

    const auto elemSize = vpux::getElemTypeSize(elementType).count();
    const auto order = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(ctx));
    const auto newStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));
    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);

    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr, /*allocSize=*/nullptr, hwSpecificFields, ctx);

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elementType, newDescAttr, memSpace,
                                                     distributedAttr, getCompressionScheme());

    const auto loc = mlir::UnknownLoc::get(getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "changeTypeComponentsForExplicitDistribution caused mismatch with quantization settings'{0}'",
                      newType);

    return newType;
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType obtained by extracting a dense tile from the original DistributedType.
// It will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offets
// adjusted for the resulting dense tile.
NDTypeInterface VPUIP::DistributedBufferType::extractDenseTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, VPU::DistributedTensorAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractDenseTile(tileOffsets, tileShape);
    }

    const auto ctx = getContext();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto compressionScheme = VPUIP::tileCompressionScheme(getCompressionScheme(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, order, getMemSpace(), distributedAttr,
                                             compressionScheme);
}

NDTypeInterface VPUIP::DistributedBufferType::extractViewTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, vpux::ShapeRef tileElemStrides,
        VPU::DistributedTensorAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractViewTile(tileOffsets, tileShape, tileElemStrides);
    }
    const auto ctx = getContext();

    const auto elemSize = getElemTypeSize().count();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto memSpace = getMemSpace();

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides();
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto newStrides = to_small_vector(tileStrides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));

    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, hwSpecificFields, ctx);

    const auto compressionScheme = VPUIP::tileCompressionScheme(getCompressionScheme(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, newDescAttr, memSpace, distributedAttr,
                                             compressionScheme);
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
    if (getCompressionScheme() != nullptr) {
        return getCompressionScheme().getTotalNumElems();
    }
    auto shape = getShape().raw();
    VPUX_THROW_UNLESS(!details::isDynamicDimValues(shape), "Cannot get element count of dynamic shaped type");
    return details::calcTotalShapeSize(shape);
}

DimsOrder VPUIP::DistributedBufferType::getDimsOrder() const {
    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

VPU::MemoryKind VPUIP::DistributedBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }

    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

Strides VPUIP::DistributedBufferType::getStrides() const {
    const auto layout = getLayout();

    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        if (auto stridesAttr = descAttr.strides()) {
            const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
            const Bit elemSize = getElemTypeSize();

            return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                               return stride * elemSize;
                                           })));
        }
    }

    // Missing strides specification means compact strides.
    const auto order = getDimsOrder();
    const auto memShape = getMemShape();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

    return order.toLogicalOrder(memStrides);
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
    auto shape = getShape();
    auto strides = getStrides();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    auto compressionScheme = getCompressionScheme();

    const auto alignStridedShape = [&](const StridedShape& stridedTiledShape) -> StridedShape {
        if (distribution.getAlignment() == nullptr) {
            return stridedTiledShape;
        }
        const auto alignment = parseIntArrayAttr<int64_t>(distribution.getAlignment());
        const auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        const auto alignedTiledShape =
                Shape(alignShape(stridedTiledShape.shape.raw(), optionalAlignment, alignValUp<int64_t>));
        const auto alignedTiledStrides =
                adaptStrides(stridedTiledShape.shape, stridedTiledShape.strides, {alignedTiledShape}, getDimsOrder());
        return StridedShape(alignedTiledShape, alignedTiledStrides.front());
    };

    const auto getAllocSize = [&](const StridedShape& stridedTiledShape, ShapeRef stridedTiledOffsets) -> Byte {
        if (compressionScheme == nullptr) {
            // the size should be calcuted base on shape and stride based on memory d0
            // if use stridedTiledShape.shape.front() and stridedTiledShape.strides.front()
            // when the order is [d0,d1,d2,d3] -> [d1,d2,d3,d0], the size is not correct.
            const auto order = getDimsOrder();
            return alignMemSize(stridedTiledShape.shape[order.dimAt(0)] * stridedTiledShape.strides[order.dimAt(0)],
                                Byte(1));
        }

        const auto axis = compressionScheme.getAxis().getInt();
        const auto numElems = compressionScheme.getNumElems().getValues<int64_t>();
        const int64_t alignment =
                (compressionScheme.getAlignment() != nullptr) ? compressionScheme.getAlignment().getInt() : 1;
        const Bit elemBitSize = getElemTypeSize();

        const auto startTileIt = numElems.begin() + stridedTiledOffsets[Dim(axis)];
        const auto endTileIt = startTileIt + stridedTiledShape.shape[Dim(axis)];
        int64_t totalBytes = 0;

        for (auto it = startTileIt; it != endTileIt; ++it) {
            auto tileByteSize = alignMemSize(elemBitSize * (*it), Byte(1)).to<Byte>().count();
            totalBytes += alignValUp<int64_t>(tileByteSize, alignment);
        }
        return Byte(totalBytes);
    };

    Byte allocSizeByte(0);

    if (distributionMode != VPU::DistributionMode::NONE) {
        const auto perClusterStridedShapes = getPerClusterMemoryStridedShapes();
        const auto perClusterOffsets = getPerClusterMemoryShapeOffsets();
        for (auto p : zip(perClusterStridedShapes, perClusterOffsets)) {
            const auto tileShape = std::get<0>(p);
            const auto tileOffsets = std::get<1>(p);
            const auto stridedTiledShape = alignStridedShape(tileShape);
            allocSizeByte = std::max(allocSizeByte, getAllocSize(stridedTiledShape, tileOffsets));
        }
    } else {
        // No distribution mode.
        Shape stridedTiledOffsets(SmallVector<int64_t>(shape.size(), 0));
        const auto stridedTiledShape = alignStridedShape(StridedShape(shape, strides));
        allocSizeByte = getAllocSize(stridedTiledShape, stridedTiledOffsets);
    }

    if (const auto memRefAttr = getLayout().dyn_cast<vpux::MemRefAttr>()) {
        auto swizzlingScheme = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        if (!swizzlingScheme || swizzlingScheme.getKey().getInt() == 0) {
            return allocSizeByte;
        }

        // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
        allocSizeByte = Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
    }

    return allocSizeByte;
}

Byte VPUIP::DistributedBufferType::getCompactAllocSize() const {
    auto shape = getShape();
    const Bit elemSize = getElemTypeSize();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    auto compressionScheme = getCompressionScheme();

    const auto alignTiledShape = [&](ShapeRef tiledShape) -> Shape {
        if (distribution.getAlignment() == nullptr) {
            return tiledShape.raw();
        }
        const auto alignment = parseIntArrayAttr<int64_t>(distribution.getAlignment());
        const auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        return Shape(alignShape(tiledShape.raw(), optionalAlignment, alignValUp<int64_t>));
    };

    const auto getAllocSize = [&](ShapeRef tiledShape, ShapeRef tiledOffsets) -> Byte {
        if (compressionScheme == nullptr) {
            return Byte(alignMemSize(elemSize * details::calcTotalShapeSize(tiledShape.raw()), Byte(1)));
        }

        const auto axis = compressionScheme.getAxis().getInt();
        const auto numElems = compressionScheme.getNumElems().getValues<int64_t>();
        const int64_t alignment =
                (compressionScheme.getAlignment() != nullptr) ? compressionScheme.getAlignment().getInt() : 1;

        const auto startTileIt = numElems.begin() + tiledOffsets[Dim(axis)];
        const auto endTileIt = startTileIt + tiledShape[Dim(axis)];
        int64_t tileElemsBytes = 0;
        for (auto it = startTileIt; it != endTileIt; ++it) {
            tileElemsBytes += alignValUp<int64_t>(*it * Byte(elemSize).count(), alignment);
        }
        return Byte(tileElemsBytes);
    };

    Byte allocSizeByte(0);

    // DUPLICATED|MULTICASTED takes priority since it means that each cluster will have the entire
    // tensor, regardless whether it's tiled or not.
    Shape tiledOffsets(SmallVector<int64_t>(shape.size(), 0));
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        const auto tiledShape = alignTiledShape(Shape(shape.raw()));
        allocSizeByte = getAllocSize(tiledShape, tiledOffsets);
    } else if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto perClusterShapes = getPerClusterMemoryShapes();
        const auto perClusterOffsets = getPerClusterMemoryShapeOffsets();
        for (auto p : zip(perClusterShapes, perClusterOffsets)) {
            const auto tileShape = std::get<0>(p);
            const auto tileOffsets = std::get<1>(p);
            const auto alignedTiledShape = alignTiledShape(tileShape);
            allocSizeByte = std::max(allocSizeByte, getAllocSize(alignedTiledShape, tileOffsets));
        }
    } else {
        // No distribution mode.
        const auto tiledShape = alignTiledShape(Shape(shape.raw()));
        allocSizeByte = getAllocSize(tiledShape, tiledOffsets);
    }

    return allocSizeByte;
}

NDTypeInterface VPUIP::DistributedBufferType::changeShape(ShapeRef shape) const {
    return changeShapeElemType(shape, getElementType());
}

NDTypeInterface VPUIP::DistributedBufferType::changeElemType(mlir::Type elemType) const {
    const auto ctx = getContext();

    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), elemType, getLayout(), getMemSpace(),
                                             getDistribution(), getCompressionScheme());
}

NDTypeInterface VPUIP::DistributedBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ctx = getContext();

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot change shape when having explicit per cluster shapes/offsets");

    const auto origOrder = getDimsOrder();
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto layoutAttr = getLayout();
    if (auto memRefAttr = getLayout().dyn_cast<vpux::MemRefAttr>()) {
        const auto orderAttr = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
        // If swizzlingKey is set get rid of strides settings
        if (memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>()) {
            layoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr,
                                               /*allocSize=*/nullptr, memRefAttr.hwSpecificFields(), ctx);
        } else {
            layoutAttr = orderAttr;
        }
    }

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, getMemSpace(),
                                                     distribution, getCompressionScheme());

    const auto loc = mlir::UnknownLoc::get(ctx);
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "ChangeShape caused mismatch with quantization settings'{0}'", newType);

    return newType;
}

NDTypeInterface VPUIP::DistributedBufferType::changeDimsOrder(DimsOrder order) const {
    const auto ctx = getContext();

    auto layoutAttr = getLayout();
    auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    if (auto memRefAttr = getLayout().dyn_cast<vpux::MemRefAttr>()) {
        // Assume compact strides
        layoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr,
                                           /*allocSize=*/nullptr, memRefAttr.hwSpecificFields(), ctx);
    } else {
        layoutAttr = orderAttr;
    }

    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), getElementType(), layoutAttr, getMemSpace(),
                                             getDistribution(), getCompressionScheme());
}

NDTypeInterface VPUIP::DistributedBufferType::changeMemSpace(IndexedSymbolAttr /*memSpace*/) const {
    VPUX_THROW("changeMemSpace method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeStrides(StridesRef strides) const {
    const auto ctx = getContext();
    const auto elemSize = getElemTypeSize().count();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto newStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));
    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, hwSpecificFields, ctx);
    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), getElementType(), newDescAttr, getMemSpace(),
                                             getDistribution(), getCompressionScheme());
}

NDTypeInterface VPUIP::DistributedBufferType::changeTypeComponents(const vpux::TypeComponents& typeComponents) const {
    const auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType());
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto strides = typeComponents.strides.value_or(getStrides());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());
    auto distribution = getDistribution();

    // If there is a shape change requested
    if (shape != Shape(getShape().toValues())) {
        VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                        "Cannot change shape when having explicit per cluster shapes/offsets");
    }

    const auto elemSize = vpux::getElemTypeSize(elementType).count();
    const auto order = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(ctx));
    const auto newStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));
    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);

    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, hwSpecificFields, ctx);

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elementType, newDescAttr, memSpace, distribution,
                                             getCompressionScheme());
}

NDTypeInterface VPUIP::DistributedBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ctx = getContext();

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot get DistributedBufferType with new shape from old one when having explicit per cluster "
                    "shapes/offsets");

    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto compressionScheme = VPUIP::tileCompressionScheme(getCompressionScheme(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, order, getMemSpace(), distribution,
                                             compressionScheme);
}

NDTypeInterface VPUIP::DistributedBufferType::extractViewTile(vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                                              vpux::ShapeRef tileElemStrides) const {
    const auto ctx = getContext();
    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot get DistributedBufferType with new shape from old one when having explicit per cluster "
                    "shapes/offsets");

    const auto elemSize = getElemTypeSize().count();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto memSpace = getMemSpace();

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides();
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto newStrides = to_small_vector(tileStrides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));

    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, hwSpecificFields, ctx);

    const auto compressionScheme = VPUIP::tileCompressionScheme(getCompressionScheme(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, newDescAttr, memSpace, distribution,
                                             compressionScheme);
}

NDTypeInterface VPUIP::DistributedBufferType::eraseTiledInfo() const {
    VPUX_THROW("eraseTiledInfo method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for DistributedBufferType");
}
