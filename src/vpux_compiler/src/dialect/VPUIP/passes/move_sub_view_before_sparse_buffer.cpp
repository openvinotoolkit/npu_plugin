//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// MoveViewOpUp
//

class MoveViewOpUp final : public mlir::OpRewritePattern<VPUIP::SubViewOp> {
public:
    MoveViewOpUp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::SubViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SubViewOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveViewOpUp::matchAndRewrite(VPUIP::SubViewOp origSubViewOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto groupSparseBuffer = origSubViewOp.source().getDefiningOp<VPUIP::GroupSparseBufferOp>();
    if (groupSparseBuffer == nullptr) {
        return mlir::failure();
    }

    // Data is mandatory, SparsityMap and Storage Element table are optional
    auto origDataValue = groupSparseBuffer.data();
    if (origDataValue == nullptr) {
        return mlir::failure();
    }
    auto sparsityMapValue = groupSparseBuffer.sparsityMap();

    VPUIP::StorageElementTableOp seTableOp = nullptr;
    if (groupSparseBuffer.storageElementTable() != nullptr) {
        seTableOp = groupSparseBuffer.storageElementTable().getDefiningOp<VPUIP::StorageElementTableOp>();
        // Do not rewrite if SETable operation is not directly attached to GroupSparseBufferOp
        if (seTableOp == nullptr) {
            return mlir::failure();
        }
    }

    auto ctx = getContext();

    const auto subViewOffsets = parseIntArrayAttr<int64_t>(origSubViewOp.static_offsets());
    const auto subViewSizes = parseIntArrayAttr<int64_t>(origSubViewOp.static_sizes());
    if (origSubViewOp.static_strides().has_value()) {
        return mlir::failure();
    }

    auto seAttr = groupSparseBuffer.seAttr().value_or(nullptr);
    auto compressionSchemeAttr = groupSparseBuffer.compression_scheme().value_or(nullptr);
    if (compressionSchemeAttr != nullptr) {
        compressionSchemeAttr =
                VPUIP::tileCompressionScheme(compressionSchemeAttr, Shape(subViewOffsets), Shape(subViewSizes));
    }

    auto rewriteInput = [&](mlir::Value value, vpux::Shape offsets, vpux::Shape sizes) {
        if (auto constOp = value.getDefiningOp<Const::DeclareOp>()) {
            // Recreate constant with subview attribute. Do not split into 2 ops otherwise when constant is
            // fused with subview then type is changed (strides are erased) without further type propagation.
            auto newContentAttr = constOp.contentAttr().subview(offsets, sizes);
            auto newConstOp = rewriter.create<Const::DeclareOp>(
                    constOp.getLoc(), vpux::convertToMemRef(newContentAttr.getType().cast<mlir::RankedTensorType>()),
                    newContentAttr);
            return newConstOp.output();
        }
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(value.getLoc(), value, getIntArrayAttr(ctx, offsets),
                                                              getIntArrayAttr(ctx, sizes));
        return newSubViewOp.result();
    };

    // Data
    auto newDataOffsets = Shape(subViewOffsets);
    auto newDataSizes = Shape(subViewSizes);
    if (seAttr != nullptr) {
        // Extract tile and get new shape for input data
        seAttr = seAttr.extractTile(Shape(subViewOffsets), Shape(subViewSizes),
                                    origDataValue.getType().cast<NDTypeInterface>().getShape(), newDataOffsets,
                                    newDataSizes);
    }
    auto newDataValue = rewriteInput(origDataValue, newDataOffsets, newDataSizes);

    // SM
    mlir::Value newSparsityMapValue = nullptr;
    if (sparsityMapValue != nullptr) {
        newSparsityMapValue = rewriteInput(sparsityMapValue, Shape(subViewOffsets), Shape(subViewSizes));
    }

    // SETable
    mlir::Value newSETableValue = nullptr;
    if (seTableOp != nullptr) {
        auto seTableOffsets = subViewOffsets;
        auto seTableSizes = subViewSizes;
        seTableOffsets[Dims4D::Act::N.ind()] = 0;
        seTableSizes[Dims4D::Act::N.ind()] = 1;

        const auto seSliceOffset = std::div(subViewOffsets[Dims4D::Act::C.ind()], seTableOp.seSize());
        VPUX_THROW_WHEN(seSliceOffset.rem != 0, "Slice over channels offset is not aligned with SE size");
        seTableOffsets[Dims4D::Act::C.ind()] = seSliceOffset.quot;

        const auto seSliceSize = std::div(subViewSizes[Dims4D::Act::C.ind()], seTableOp.seSize());
        VPUX_THROW_WHEN(seSliceSize.rem != 0, "Slice over channels size is not aligned with SE size");
        seTableSizes[Dims4D::Act::C.ind()] = seSliceSize.quot;

        auto seTableSliceOp = rewriter.create<VPUIP::SubViewOp>(origSubViewOp.getLoc(), seTableOp.output(),
                                                                getIntArrayAttr(ctx, seTableOffsets),
                                                                getIntArrayAttr(ctx, seTableSizes));
        newSETableValue = seTableSliceOp.result();
    }

    auto newOp = rewriter.replaceOpWithNewOp<VPUIP::GroupSparseBufferOp>(
            origSubViewOp, newDataValue, newSparsityMapValue, newSETableValue, groupSparseBuffer.is_weightsAttr(),
            compressionSchemeAttr, seAttr);

    // Reinfer child ops return types if necessary due to strides info may be erased
    auto currentOp = newOp.getOperation();
    while (currentOp != nullptr) {
        if (mlir::isa<mlir::InferTypeOpInterface>(currentOp)) {
            vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
        }
        currentOp = currentOp->getNextNode();
    }

    return mlir::success();
}

//
// MoveSubViewBeforeSparseBufferPass
//

class MoveSubViewBeforeSparseBufferPass final :
        public VPUIP::MoveSubViewBeforeSparseBufferBase<MoveSubViewBeforeSparseBufferPass> {
public:
    explicit MoveSubViewBeforeSparseBufferPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveSubViewBeforeSparseBufferPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveViewOpUp>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveSubViewBeforeSparseBufferPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveSubViewBeforeSparseBufferPass(Logger log) {
    return std::make_unique<MoveSubViewBeforeSparseBufferPass>(log);
}
