//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

//
// Builders
//

void vpux::VPUIP::StorageElementTableOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                               ArrayRef<int64_t> dataShape, mlir::Type dataElemType, int64_t seSize,
                                               int64_t seDepth, VPU::SEAttr seAttr) {
    auto dataShapeAttr = getIntArrayAttr(odsBuilder.getContext(), dataShape);
    build(odsBuilder, odsState, dataShapeAttr, dataElemType, seSize, seDepth, seAttr, nullptr, nullptr);
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPUIP::StorageElementTableOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::StorageElementTableOpAdaptor setOp(operands, attrs);
    if (mlir::failed(setOp.verify(loc))) {
        return mlir::failure();
    }

    const auto depth = setOp.seDepth();
    const auto dataShape = parseIntArrayAttr<int64_t>(setOp.dataShape());
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected 4D input data, got {0} dimensions", dataShape.size());
    Shape shapeAfterSERead(dataShape);
    if (auto seAttrValue = setOp.seAttr().value_or(nullptr)) {
        shapeAfterSERead = seAttrValue.inferOutputShape(shapeAfterSERead);
    }
    const auto height = shapeAfterSERead[Dims4D::Act::H];
    const auto width = shapeAfterSERead[Dims4D::Act::W];
    SmallVector<int64_t> shape{1, depth, height, width};

    const auto outType = getMemRefType(ShapeRef(shape), getInt32Type(ctx), DimsOrder::NHWC, /*memSpace=*/nullptr);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPUIP::StorageElementTableOp::verify() {
    const auto setOp = getOperation();
    using namespace VPU::NCESparsity;

    if (auto seAttrValue = seAttr().value_or(nullptr)) {
        VPUX_THROW_WHEN(seAttrValue.isa<VPU::SEInterpolateAttr>() == false,
                        "Only SEInterpolateAttr is supported for VPU::SEAttr");
    }

    if (!basePtrs().has_value()) {
        return mlir::success();
    }

    const auto opBasePtrs = basePtrs().value().getValues<int32_t>();
    const auto expectedNumPtrs = output().getType().cast<vpux::NDTypeInterface>().getNumElements();
    if (static_cast<size_t>(expectedNumPtrs) != opBasePtrs.size()) {
        return errorAt(setOp->getLoc(), "StorageElementTable expects to have {0}, but got {1}", expectedNumPtrs,
                       opBasePtrs.size());
    }

    const mlir::DenseSet<int32_t> distinctPtrs(opBasePtrs.begin(), opBasePtrs.end());
    if (distinctPtrs.size() > MAX_DISTINCT_BASE_PTRS) {
        return errorAt(setOp->getLoc(),
                       "StorageElementTable expects to have at most {0} unique values for base_ptrs, but have {1}.",
                       MAX_DISTINCT_BASE_PTRS, distinctPtrs.size());
    }

    const auto allPtrsAreValid = llvm::all_of(opBasePtrs, [](auto ptr) {
        return 0 <= ptr && ptr <= MAX_BASE_POINTER_VALUE;
    });
    if (!allPtrsAreValid) {
        return errorAt(setOp->getLoc(), "Operation contains invalid base_ptrs");
    }

    return mlir::success();
}

//
// Canonicalizers
//

namespace {
class FuseChildSubviewOps final : public mlir::OpRewritePattern<VPUIP::StorageElementTableOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::StorageElementTableOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseChildSubviewOps::matchAndRewrite(VPUIP::StorageElementTableOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto seAttr = origOp.seAttr().value_or(nullptr);
    if (seAttr == nullptr) {
        return mlir::failure();
    }
    for (auto userOp : llvm::make_early_inc_range(origOp.output().getUsers())) {
        if (auto subViewUserOp = mlir::dyn_cast<VPUIP::SubViewOp>(userOp)) {
            const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewUserOp.static_offsets());
            const auto subViewSizes = parseIntArrayAttr<int64_t>(subViewUserOp.static_sizes());
            const auto subViewStrides = subViewUserOp.static_strides();
            VPUX_THROW_WHEN(subViewStrides.has_value(),
                            "Strides are not supported for SubView of StorageElementTableOp");

            auto effectiveOutputOffsets = subViewOffsets;
            auto effectiveOutputSizes = subViewSizes;
            effectiveOutputOffsets[Dims4D::Act::C.ind()] *= origOp.seSize();
            effectiveOutputSizes[Dims4D::Act::C.ind()] *= origOp.seSize();

            const auto inputDataShape = Shape(parseIntArrayAttr<int64_t>(origOp.dataShape()));
            auto inputTileShape = Shape(inputDataShape.size());
            auto inputTileOffset = inputTileShape;
            seAttr = seAttr.extractTile(Shape(effectiveOutputOffsets), Shape(effectiveOutputSizes), inputDataShape,
                                        inputTileOffset, inputTileShape);
            auto newSETableOp = rewriter.replaceOpWithNewOp<VPUIP::StorageElementTableOp>(
                    subViewUserOp, inputTileShape.raw(), origOp.dataElemType(), origOp.seSize(),
                    subViewSizes[Dims4D::Act::C.ind()], seAttr);
            auto currentOp = newSETableOp.getOperation();
            while (currentOp != nullptr) {
                if (mlir::isa<mlir::InferTypeOpInterface>(currentOp)) {
                    vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
                }
                currentOp = currentOp->getNextNode();
            }
        }
    }
    return mlir::success();
}

}  // namespace

void vpux::VPUIP::StorageElementTableOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                                                     mlir::MLIRContext* ctx) {
    results.add<FuseChildSubviewOps>(ctx);
}
