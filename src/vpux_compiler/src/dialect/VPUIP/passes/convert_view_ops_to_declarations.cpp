//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

//
// ViewLikeRewrite
//

class ViewLikeRewrite final : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(mlir::MLIRContext* ctx, const AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _aliasInfo(aliasInfo), _log(log) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte calculateOffset(mlir::Value val) const;

private:
    const AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

Byte ViewLikeRewrite::calculateOffset(mlir::Value val) const {
    Byte offset(0);

    if (auto source = _aliasInfo->getSource(val)) {
        offset = calculateOffset(source);
    }

    if (auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(val.getDefiningOp())) {
        offset += Byte(declareOp.getByteOffset());
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(val.getDefiningOp())) {
        auto strides = getStrides(subViewOp.getSource());
        const auto offsets = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
        VPUX_THROW_UNLESS(strides.size() == offsets.size(), "SubView offsets '{0}' doesn't match strides '{1}'",
                          offsets, strides);

        auto distributedType = subViewOp.getSource().getType().dyn_cast<VPUIP::DistributedBufferType>();

        VPU::DistributedTensorAttr distribution;
        std::optional<int64_t> tileIndex;
        int64_t numTile = 0, tileIndexVal = 0;
        if (distributedType != nullptr) {
            distribution = distributedType.getDistribution();
            tileIndex = VPUIP::getTilingDimIndex(distributedType);
            if (tileIndex.has_value()) {
                tileIndexVal = tileIndex.value();
                numTile = parseIntArrayAttr<int64_t>(distribution.getNumTiles())[tileIndexVal];
            }
        }

        auto getSameAxisForClusterTiledSlice = [&]() -> std::optional<int64_t> {
            if (!tileIndex.has_value()) {
                return std::nullopt;
            }

            auto origShape = getShape(subViewOp.getSource());
            auto subShape = getShape(subViewOp.getResult());
            if (origShape.size() != 4 || origShape.size() != subShape.size()) {
                return std::nullopt;
            }

            // ClusterTiling and subview are done on the same axis
            if (origShape[Dim(tileIndexVal)] != subShape[Dim(tileIndexVal)]) {
                VPUX_THROW_WHEN(distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED,
                                "Cannot extract correct address for subview with OVERLAPPED distribution mode and "
                                "subview axis same as clustering axis");
                return tileIndexVal;
            }

            return std::nullopt;
        };

        const auto sameClusteringSliceAxis = getSameAxisForClusterTiledSlice();

        // update strides based on numTiles
        if (distributedType && distribution.getMode().getValue() == VPU::DistributionMode::SEGMENTED) {
            // The algorithm for sameClusteringSlice does not need updated strides
            if (!sameClusteringSliceAxis.has_value() && tileIndex.has_value()) {
                auto dimOrder = DimsOrder::fromValue(val);
                const auto origShape = getShape(subViewOp.getSource());
                const auto tiledShape = divUp(origShape[Dim(tileIndex.value())], numTile);
                const auto tiledMemAxis = dimOrder.dimPos(Dim(tileIndex.value()));
                auto permutation =
                        to_small_vector(distributedType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                            return checked_cast<uint32_t>(dim.ind());
                                        }));

                for (int64_t i = static_cast<int64_t>(tiledMemAxis) - 1; i >= 0; --i) {
                    auto curDim = Dim(permutation[i]);
                    auto lowerDim = Dim(permutation[i + 1]);
                    if (i == static_cast<int64_t>(tiledMemAxis) - 1) {
                        strides[curDim] = strides[lowerDim] * tiledShape;
                    } else {
                        strides[curDim] = strides[lowerDim] * origShape[lowerDim];
                    }
                }
            }
        }

        for (int64_t axis = 0; axis < static_cast<int64_t>(strides.size()); axis++) {
            const auto stride = strides[Dim(axis)];
            const auto sliceOffset = offsets[axis];

            if (sameClusteringSliceAxis.has_value() && sameClusteringSliceAxis.value() == axis) {
                // When clustering axis is the same as Subview axis, the offsets are relative to the full un-clustered
                // buffer. We make the assumption that the offset to current slice is distributed equally across
                // clusters.
                // E.g.:
                // VPUIP.SubView %source [0, 0, 0, 0] [1, 12, 186, 240] -> SEGMENTED with numTiles = [1, 1, 4, 1]
                // 0 - offset in orig shape, divided into 4 clusters
                //          => subview_start_offset = stride * (0 / 4) = 0 ~ offset0
                // VPUIP.SubView %source [0, 0, 186, 0] [1, 12, 186, 240] -> SEGMENTED with numTiles = [1, 1, 4, 1]
                // 186 - offset in orig shape, divided into 4 clusters
                //          => subview_start_offset = stride * divUp(186, 4) = stride * 47 ~ offset1
                // The distribution in memory for this example would be:
                //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
                // offset0  x_______________________________________________________________
                //          |  47 lines of  |  47 lines of  |  46 lines of  |  46 lines of  |
                //          | actual data   | actual data   | actual data   | actual data   |
                //          |               |               |---------------|---------------|
                // offset1  x---------------|---------------|---------------|---------------|
                //          |  47 lines of  |    47 lines   |    46 lines   |    46 lines   |
                //          | actual data   |               |---------------|---------------|
                //          |_______________|_______________|_______________|_______________|

                // TODO: Above scenario happens mostly in the context of act shave tiling and are subject to the
                // following assumptions: SEGMENTED distribution mode, 2 Act Shaves/per cluster
                // Clean up ticket: E#98440
                offset += Byte(stride * divUp(sliceOffset, numTile));
            } else {
                // Compute simple offset
                offset += Byte(stride * sliceOffset);
            }
        }
    }

    return offset;
}

mlir::LogicalResult ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp,
                   VPUIP::DistributedCastOp, VPUIP::ShapeCastOp, VPUIP::StubOp, VPUIP::ViewOp, VPUIP::WorkloadCastOp>(
                origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origVal = origOp->getResult(0);
    const Byte offset = calculateOffset(origVal);

    const auto roots = _aliasInfo->getRoots(origVal);
    VPUX_THROW_UNLESS(roots.size() == 1, "Value '{0}' expected to have only one root. Got {1}", origVal, roots.size());
    const auto rootVal = *roots.begin();

    VPURT::BufferSection section = VPURT::BufferSection::DDR;
    std::optional<mlir::ArrayAttr> sectionIndex;

    if (auto declareOp = rootVal.getDefiningOp<VPURT::DeclareBufferOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}'", declareOp->getLoc());

        const auto outType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        section = VPURT::symbolizeBufferSection(outType.getMemSpace().getLeafName()).value();
        auto memSpaceIndex = outType.getMemSpace().getIndex();
        if (memSpaceIndex.has_value()) {
            sectionIndex = getIntArrayAttr(rewriter, ArrayRef({memSpaceIndex.value()}));
        }
    } else if (auto blockArg = rootVal.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases Block argument '{0}'", blockArg);

        auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
        VPUX_THROW_UNLESS(funcOp != nullptr, "The view source doesn't belong to Function");

        const auto argInd = checked_cast<size_t>(blockArg.getArgNumber());

        const auto numOutputs = funcOp.getNumResults();
        VPUX_THROW_UNLESS(numOutputs < funcOp.getNumArguments(), "The Function '@{0}' is not bufferized",
                          funcOp.getName());

        size_t numProfilingOutputs = 0;
        if (auto module = blockArg.getParentRegion()->getParentOfType<mlir::ModuleOp>()) {
            auto netOps = to_small_vector(module.getOps<IE::CNNNetworkOp>());
            if (!netOps.empty()) {
                numProfilingOutputs = netOps.front().getProfilingOutputsCount();
            }
        }
        const auto numNetOutputs = numOutputs - numProfilingOutputs;
        const auto numNetInputs = funcOp.getNumArguments() - numOutputs;

        int64_t sectionIndexVal;
        if (argInd < numNetInputs) {
            _log.nest(2).trace("It aliases network input");

            section = VPURT::BufferSection::NetworkInput;
            sectionIndexVal = argInd;
        } else if (argInd < numNetInputs + numNetOutputs) {
            _log.nest(2).trace("It aliases network output");

            section = VPURT::BufferSection::NetworkOutput;
            sectionIndexVal = argInd - numNetInputs;
        } else if (argInd < numNetInputs + numOutputs) {
            _log.nest(2).trace("It aliases network output");

            section = VPURT::BufferSection::ProfilingOutput;
            sectionIndexVal = argInd - numNetInputs - numNetOutputs;
        } else {
            VPUX_THROW("The view source doesn't belong to network entry point Function");
        }
        sectionIndex = getIntArrayAttr(getContext(), ArrayRef(sectionIndexVal));
    } else {
        VPUX_THROW("Unknown source owner");
    }

    const auto outType = origOp->getResult(0).getType();
    auto swizzlingScheme = getSwizzlingSchemeAttr(outType);
    mlir::IntegerAttr swizzlingKey;
    if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
        swizzlingKey = swizzlingScheme.getKey();
    }

    mlir::ArrayAttr sectionIndexAttr = sectionIndex.has_value() ? sectionIndex.value() : nullptr;
    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, outType, section, sectionIndexAttr, offset.count(),
                                                        swizzlingKey);

    return mlir::success();
}

class RewriteConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    RewriteConcatView(::mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RewriteConcatView::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    for (auto input : origOp.getInputs()) {
        if (auto waitOp = input.getDefiningOp<mlir::async::AwaitOp>()) {
            if (waitOp->hasOneUse()) {
                waitOp->dropAllUses();
                waitOp->erase();
            }
        }
    }

    rewriter.replaceOp(origOp, origOp.getOutputBuff());
    return ::mlir::success();
}

//
// ConvertViewOpsToDeclarationsPass
//

class ConvertViewOpsToDeclarationsPass final :
        public VPUIP::ConvertViewOpsToDeclarationsBase<ConvertViewOpsToDeclarationsPass> {
public:
    explicit ConvertViewOpsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertViewOpsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    target.addIllegalOp<VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::ConcatViewOp, VPUIP::PermuteCastOp,
                        VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp, VPUIP::ShapeCastOp, VPUIP::StubOp,
                        VPUIP::ViewOp, VPUIP::WorkloadCastOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ViewLikeRewrite>(&ctx, &aliasInfo, _log);
    patterns.add<RewriteConcatView>(&ctx);

    auto func = getOperation();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertViewOpsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertViewOpsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertViewOpsToDeclarationsPass>(log);
}
