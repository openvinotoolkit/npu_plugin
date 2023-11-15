//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// build
//

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::SliceOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SliceOpAdaptor sliceOp(operands, attrs);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = sliceOp.source().getType().dyn_cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return errorAt(loc, "VPU::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    auto inferExplicitDistributedAttr = [&](VPU::DistributedTensorType distributedIn) -> VPU::DistributedTensorAttr {
        const auto origDistribution = distributedIn.getDistribution();
        const auto inShape = distributedIn.getShape().raw();

        if (origDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, sliceShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, sliceShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Slice Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
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

    auto distributedIn = origType.dyn_cast<VPU::DistributedTensorType>();
    if (distributedIn != nullptr &&
        VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedIn.getDistribution())) {
        const auto sliceDistributedAttr = inferExplicitDistributedAttr(distributedIn);
        const auto newType = distributedIn.extractDenseTileForExplicitDistribution(
                ShapeRef(sliceOffsets), ShapeRef(sliceShape), sliceDistributedAttr);
        inferredTypes.emplace_back(newType);
    } else {
        const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
        inferredTypes.emplace_back(newType);
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult VPU::SliceOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (source().getType() == result().getType()) {
        return source();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(static_offsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(static_sizes()));
        return origContent.subview(offset, shape);
    }

    return nullptr;
}

//
// ComposeSlice
//

namespace {

class ComposeSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSlice::matchAndRewrite(VPU::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.source().getDefiningOp<VPU::SliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.static_offsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.static_offsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.static_sizes();
    rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, producerSliceOp.source(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSlice>(ctx);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SliceOp::serialize(EMU::BlobWriter& writer) {
    const auto begin = writer.createVector(parseIntArrayAttr<uint32_t>(static_offsets()));
    const auto size = writer.createVector(parseIntArrayAttr<uint32_t>(static_sizes()));

    MVCNN::SliceParamsBuilder builder(writer);
    builder.add_begin(begin);
    builder.add_size(size);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SliceParams});
}
