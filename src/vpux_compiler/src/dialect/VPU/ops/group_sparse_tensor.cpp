//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

//
// build
//

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           bool isWeights, VPU::CompressionSchemeAttr compressionScheme) {
    build(builder, state, data, nullptr, nullptr, isWeights, compressionScheme);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, bool isWeights,
                                           VPU::CompressionSchemeAttr compressionScheme) {
    build(builder, state, data, sparsityMap, nullptr, isWeights, compressionScheme);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable, bool isWeights,
                                           VPU::CompressionSchemeAttr compressionScheme) {
    const auto isWeightsAttr = isWeights ? mlir::UnitAttr::get(builder.getContext()) : nullptr;
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, compressionScheme, nullptr);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable,
                                           VPU::SEAttr seAttr) {
    build(builder, state, data, sparsityMap, storageElementTable, nullptr, nullptr, seAttr);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::GroupSparseTensorOp::inferReturnTypes(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange /*ranges*/, SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupSparseTensorOpAdaptor groupOp(operands, attrs);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.data().getType();
    const auto sparsityMapType = groupOp.sparsityMap() != nullptr ? groupOp.sparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.storageElementTable() != nullptr ? groupOp.storageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType,
                                                             groupOp.is_weightsAttr(), groupOp.compression_schemeAttr(),
                                                             groupOp.seAttrAttr()));

    return mlir::success();
}

//
// MoveViewLikeOps
//

/*
 * Patterns such as the following:
 *      Data   Const SM   SETable
 *        \       |       /
 *        GroupSparseTensor
 *       /              \
 *     Slice           Slice
 *
 * get transformed into:
 *      Data    Const SM*   SETable     Const Data  Const SM* SETable
 *        |        |         |             |           |       |
 *      Slice      |       Slice         Slice         |      Slice
 *        \        |        /               \          |      /
 *         GroupSparseTensor                 GroupSparseTensor
 *
 * This can allow the Slice canonicalizer convert the operation into a constant transformation.
 * The sparsity map for weights is attached directly as a transformation since the original constant's type has the
 * shape flattened for each output channel (i.e. OCx1x1xSIZExi1), making it incompatible with the attributes of the
 * Slice operation. Therefore, it is applied as a transformation to the dense constant before the
 * transformation that generates the sparsity map.
 * StorageElementTableOp has its own canonicalizer and SliceOp will be fused into it.
 */

namespace {

class MoveViewLikeOps final : public mlir::OpRewritePattern<VPU::GroupSparseTensorOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MoveViewLikeOps::matchAndRewrite(VPU::GroupSparseTensorOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto origDataValue = origOp.data();
    if (origDataValue == nullptr) {
        return mlir::failure();
    }
    auto sparsityMapValue = origOp.sparsityMap();

    VPU::StorageElementTableOp seTableOp = nullptr;
    if (origOp.storageElementTable() != nullptr) {
        seTableOp = origOp.storageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
        if (seTableOp == nullptr) {
            return mlir::failure();
        }
    }

    auto ctx = getContext();
    for (auto userOp : llvm::make_early_inc_range(origOp.output().getUsers())) {
        if (auto sliceUserOp = mlir::dyn_cast<VPU::SliceOp>(userOp)) {
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceUserOp.static_offsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceUserOp.static_sizes());

            auto seAttr = origOp.seAttr().value_or(nullptr);
            auto compressionSchemeAttr = origOp.compression_scheme().value_or(nullptr);
            if (compressionSchemeAttr != nullptr) {
                compressionSchemeAttr =
                        VPU::tileCompressionScheme(compressionSchemeAttr, Shape(sliceOffsets), Shape(sliceSizes));
            }

            auto rewriteInput = [&](mlir::Value value, vpux::Shape offsets, vpux::Shape sizes) {
                if (auto constOp = value.getDefiningOp<Const::DeclareOp>()) {
                    auto newContentAttr = constOp.contentAttr().subview(offsets, sizes);
                    auto newConstOp = rewriter.create<Const::DeclareOp>(constOp.getLoc(), newContentAttr.getType(),
                                                                        newContentAttr);
                    return newConstOp.output();
                }
                auto newSliceOp = rewriter.create<VPU::SliceOp>(value.getLoc(), value, getIntArrayAttr(ctx, offsets),
                                                                getIntArrayAttr(ctx, sizes));
                return newSliceOp.result();
            };

            // Data
            auto newDataOffsets = Shape(sliceOffsets);
            auto newDataSizes = Shape(sliceSizes);
            if (seAttr != nullptr) {
                seAttr = seAttr.extractTile(Shape(sliceOffsets), Shape(sliceSizes),
                                            origDataValue.getType().cast<NDTypeInterface>().getShape(), newDataOffsets,
                                            newDataSizes);
            }
            auto newDataValue = rewriteInput(origDataValue, newDataOffsets, newDataSizes);

            // SM
            mlir::Value newSparsityMapValue = nullptr;
            if (sparsityMapValue != nullptr) {
                newSparsityMapValue = rewriteInput(sparsityMapValue, Shape(sliceOffsets), Shape(sliceSizes));
            }

            // SETable
            mlir::Value newSETableValue = nullptr;
            if (seTableOp != nullptr) {
                auto seTableOffsets = sliceOffsets;
                auto seTableSizes = sliceSizes;
                seTableOffsets[Dims4D::Act::N.ind()] = 0;
                seTableSizes[Dims4D::Act::N.ind()] = 1;

                const auto seSliceOffset = std::div(sliceOffsets[Dims4D::Act::C.ind()], seTableOp.seSize());
                VPUX_THROW_WHEN(seSliceOffset.rem != 0, "Slice over channels offset is not aligned with SE size");
                seTableOffsets[Dims4D::Act::C.ind()] = seSliceOffset.quot;

                const auto seSliceSize = std::div(sliceSizes[Dims4D::Act::C.ind()], seTableOp.seSize());
                VPUX_THROW_WHEN(seSliceSize.rem != 0, "Slice over channels size is not aligned with SE size");
                seTableSizes[Dims4D::Act::C.ind()] = seSliceSize.quot;

                auto seTableSliceOp = rewriter.create<VPU::SliceOp>(sliceUserOp.getLoc(), origOp.storageElementTable(),
                                                                    getIntArrayAttr(ctx, seTableOffsets),
                                                                    getIntArrayAttr(ctx, seTableSizes));
                newSETableValue = seTableSliceOp.result();
            }
            rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(sliceUserOp, newDataValue, newSparsityMapValue,
                                                                  newSETableValue, origOp.is_weightsAttr(),
                                                                  compressionSchemeAttr, seAttr);
        }
    }

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::GroupSparseTensorOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                                                 mlir::MLIRContext* ctx) {
    results.add<MoveViewLikeOps>(ctx);
}
