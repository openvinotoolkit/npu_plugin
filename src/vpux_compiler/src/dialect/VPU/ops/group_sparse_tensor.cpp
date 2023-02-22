//
// Copyright (C) 2022 Intel Corporation.
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
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, compressionScheme);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::GroupSparseTensorOp::inferReturnTypes(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange /*ranges*/, SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GroupSparseTensorOpAdaptor groupOp(operands, attrs);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.data().getType();
    const auto sparsityMapType = groupOp.sparsityMap() != nullptr ? groupOp.sparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.storageElementTable() != nullptr ? groupOp.storageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType,
                                                             groupOp.is_weightsAttr(),
                                                             groupOp.compression_schemeAttr()));

    return mlir::success();
}

//
// MoveConstantViewLikeOps
//

namespace {

class MoveConstantViewLikeOps final : public mlir::OpRewritePattern<VPU::GroupSparseTensorOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp op, mlir::PatternRewriter& rewriter) const final;
};

/*
 * Patterns such as the following:
 *    Const Data    Const SM
 *        \           /
 *      GroupSparseTensor
 *       /             \
 *     Slice         Slice
 *
 * get transformed into:
 *    Const Data    Const SM*      Const Data    Const SM*
 *        |            |              |            |
 *      Slice          |            Slice          |
 *        \            /              \            /
 *      GroupSparseTensor            GroupSparseTensor
 *
 * This can allow the Slice canonicalizer convert the operation into a constant transformation for the data.
 * The sparsity map is attached directly as a transformation since the original constant's type has the shape
 * flattened for each output channel (i.e. OCx1x1xSIZExi1), making it incompatible with the attributes of the
 * Slice operation. Therefore, it is applied as a transformation to the dense constant before the
 * transformation that generates the sparsity map.
 */
mlir::LogicalResult MoveConstantViewLikeOps::matchAndRewrite(VPU::GroupSparseTensorOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto constDataOp = origOp.data().getDefiningOp<Const::DeclareOp>();
    if (constDataOp == nullptr) {
        return mlir::failure();
    }

    Const::DeclareOp constSparsityMapOp = nullptr;
    if (origOp.sparsityMap() != nullptr) {
        constSparsityMapOp = origOp.sparsityMap().getDefiningOp<Const::DeclareOp>();
        if (constSparsityMapOp == nullptr) {
            return mlir::failure();
        }
    }

    for (auto userOp : llvm::make_early_inc_range(origOp.output().getUsers())) {
        if (auto sliceUserOp = mlir::dyn_cast<VPU::SliceOp>(userOp)) {
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceUserOp.static_offsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceUserOp.static_sizes());

            auto newConstDataOp = rewriter.clone(*constDataOp);
            auto dataSliceOp =
                    rewriter.create<VPU::SliceOp>(sliceUserOp.getLoc(), newConstDataOp->getResult(0),
                                                  sliceUserOp.static_offsetsAttr(), sliceUserOp.static_sizesAttr());
            Const::DeclareOp sparsityMapOp = nullptr;
            if (constSparsityMapOp != nullptr) {
                const auto contentAttr = constSparsityMapOp.contentAttr();
                const auto newContentAttr = contentAttr.subview(Shape(sliceOffsets), Shape(sliceSizes));

                sparsityMapOp = rewriter.create<Const::DeclareOp>(constSparsityMapOp.getLoc(), newContentAttr.getType(),
                                                                  newContentAttr);
            }

            auto compressionScheme = origOp.compression_schemeAttr();
            compressionScheme = VPU::tileCompressionScheme(compressionScheme, Shape(sliceOffsets), Shape(sliceSizes));

            rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(
                    sliceUserOp, dataSliceOp.result(), sparsityMapOp.output(), nullptr,
                    origOp.is_weights().getValueOr(false), compressionScheme);
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
    results.add<MoveConstantViewLikeOps>(ctx);
}
