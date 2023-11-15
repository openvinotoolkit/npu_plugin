//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes) {
    build(builder, state, input, static_offsets, static_sizes, nullptr);
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes, ShapeRef static_strides) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw(), static_strides.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes,
                             ArrayRef<int64_t> static_strides) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes), getIntArrayAttr(builder.getContext(), static_strides));
}

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::SubViewOp::getViewSource() {
    return source();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::SubViewOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::SubViewOpAdaptor subViewOp(operands, attrs);
    if (mlir::failed(subViewOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = subViewOp.source().getType().cast<NDTypeInterface>();

    const auto subViewShape = parseIntArrayAttr<int64_t>(subViewOp.static_sizes());
    const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.static_offsets());
    const auto subViewStrides = subViewOp.static_strides().has_value()
                                        ? parseIntArrayAttr<int64_t>(subViewOp.static_strides().value())
                                        : SmallVector<int64_t>(origType.getRank(), 1);

    if (subViewShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile shape '{0}' doesn't match MemRef rank '{1}'", subViewShape, origType.getRank());
    }
    if (subViewOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile offsets '{0}' doesn't match MemRef rank '{1}'", subViewOffsets, origType.getRank());
    }
    if (subViewStrides.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile strides '{0}' doesn't match MemRef rank '{1}'", subViewStrides, origType.getRank());
    }

    auto inferExplicitDistributedAttr = [&](VPUIP::DistributedBufferType distributedIn) -> VPU::DistributedTensorAttr {
        const auto origDistribution = distributedIn.getDistribution();
        const auto inShape = distributedIn.getShape().raw();

        if (origDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, subViewShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, subViewShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Subview Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (subViewShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = subViewShape[dim];
                }
            }
        }
        const auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);

        return VPU::DistributedTensorAttr::get(
                ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(), perClusterShapesAttr,
                origDistribution.getMemoryOffsets(), perClusterShapesAttr, origDistribution.getMemoryOffsets(),
                origDistribution.getEqualMemoryAndComputeView());
    };

    auto distributedIn = origType.dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedIn != nullptr &&
        VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedIn.getDistribution())) {
        const auto subViewDistributedAttr = inferExplicitDistributedAttr(distributedIn);
        const auto subViewType = distributedIn.extractViewTileForExplicitDistribution(
                ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides), subViewDistributedAttr);
        inferredTypes.push_back(subViewType);
    } else {
        const auto subViewType =
                origType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides));

        inferredTypes.push_back(subViewType);
    }

    return mlir::success();
}

// A sparsity map constant has the workload size flattened, so that its shape is OCx1x1xSIZE.
// Therefore, only subviews over the OC dimension are allowed.
// Additionally, the GetSparsityMap transformation is the last one in the list. When folding
// subviews into the constant, it will be introduced as a transformation before it, so its
// subview dimensions have to be adapted for the shape before flattening.
void adaptSparsityMapConstant(mlir::Value source, Shape& offset, Shape& shape) {
    auto constParentOp = source.getDefiningOp<Const::DeclareOp>();
    if (constParentOp == nullptr) {
        return;
    }
    const auto transformations = constParentOp.getContentAttr().getTransformations();
    if (transformations.empty()) {
        return;
    }

    auto getSparistyMapTransIt = std::find_if(transformations.rbegin(), transformations.rend(),
                                              [&](vpux::Const::TransformAttrInterface trans) {
                                                  return trans.isa<Const::GetSparsityMapAttr>();
                                              });
    if (getSparistyMapTransIt == transformations.rend()) {
        return;
    }

    auto posFromEnd = std::distance(transformations.rbegin(), getSparistyMapTransIt);

    const auto zeroWorkloadOffsets = std::all_of(offset.begin() + 1, offset.end(), [](const int64_t value) {
        return value == 0;
    });
    VPUX_THROW_UNLESS(zeroWorkloadOffsets, "Offsets with non-zero values for workloads are not supported. Got {0}",
                      offset);

    const auto sparsityMapShape = constParentOp.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto sparsityMapWorkloadShape = SmallVector<int64_t>(sparsityMapShape.begin() + 1, sparsityMapShape.end());
    const auto shapeWorkloadShape = SmallVector<int64_t>(shape.begin() + 1, shape.end());
    for (auto p : zip(sparsityMapWorkloadShape, shapeWorkloadShape)) {
        const auto sparsityMapDim = std::get<0>(p);
        const auto shapeDim = std::get<1>(p);
        VPUX_THROW_UNLESS(sparsityMapDim == shapeDim,
                          "Subview shape with different workload size is not supported: original dim {0}, new dim {1}",
                          sparsityMapDim, shapeDim);
    }

    auto inputType = constParentOp.getContentAttr().getBaseContent().getType().cast<vpux::NDTypeInterface>();
    for (auto idx : irange(transformations.size() - (1 + posFromEnd))) {
        inputType = transformations[idx].inferOutputType(inputType);
    }
    const auto inputShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected a 4-dimensional type, got {0} dimensions", inputShape.size());
    const auto OC = shape.raw()[0];
    shape = Shape({OC, inputShape[1], inputShape[2], inputShape[3]});
}

//
// fold
//

mlir::OpFoldResult VPUIP::SubViewOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (source().getType() == result().getType()) {
        return source();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        auto offset = Shape(parseIntArrayAttr<int64_t>(static_offsets()));
        auto shape = Shape(parseIntArrayAttr<int64_t>(static_sizes()));
        adaptSparsityMapConstant(source(), offset, shape);
        return origContent.subview(offset, shape);
    }

    return nullptr;
}

//
// ComposeSubView
//

namespace {

class ComposeSubView final : public mlir::OpRewritePattern<VPUIP::SubViewOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::SubViewOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSubView::matchAndRewrite(VPUIP::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.source().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.static_strides().has_value() || producerSubViewOp.static_strides().has_value()) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSubViewOp.static_offsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.static_offsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.static_sizes();
    rewriter.replaceOpWithNewOp<VPUIP::SubViewOp>(origOp, producerSubViewOp.source(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::SubViewOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSubView>(ctx);
}
