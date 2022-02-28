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

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

VPU::MPEMode getMpeModeForKmb(mlir::Type inElemType, mlir::Type outElemType, mlir::Operation*) {
    if (inElemType.isa<mlir::quant::QuantizedType>() || outElemType.isa<mlir::quant::QuantizedType>()) {
        return VPU::MPEMode::VECTOR;
    }

    if (inElemType.isF16() || inElemType.isBF16() || outElemType.isF16() || outElemType.isBF16()) {
        return VPU::MPEMode::VECTOR_FP16;
    }

    // Let's fall back to vector (might be a bad idea though).
    return VPU::MPEMode::VECTOR;
}

VPU::MPEMode getMpeModeForMtl(mlir::Type, mlir::Type, mlir::Operation* operation) {
    if (mlir::isa<VPU::NCEConvolutionOp>(operation)) {
        return VPU::MPEMode::CUBOID_16x16;
    } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(operation) || mlir::isa<VPU::NCEMaxPoolOp>(operation)) {
        return VPU::MPEMode::CUBOID_4x16;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(operation)) {
        return VPU::MPEMode::CUBOID_8x16;
    }

    return VPU::MPEMode::CUBOID_16x16;
}

using GetMpeModeCb = VPU::MPEMode (*)(mlir::Type, mlir::Type, mlir::Operation*);

const EnumMap<VPU::ArchKind, GetMpeModeCb> mpeMap = {
        {VPU::ArchKind::KMB, getMpeModeForKmb},
        {VPU::ArchKind::TBH, getMpeModeForKmb},
        {VPU::ArchKind::MTL, getMpeModeForMtl},
};

//
// GenericNCERewrite
//

template <class ConcreteOp>
class GenericNCERewrite final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    Logger _log;
};

void addDPUTasks(mlir::PatternRewriter& rewriter, VPU::NCEOpInterface origOp, VPU::PaddingAttr pads,
                 const mlir::Type inElemType, const mlir::Type outElemType, ShapeRef outputShape, int64_t numDPU,
                 VPU::ArchKind arch) {
    const auto mpeByType = mpeMap.at(arch);

    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = outputShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(outputShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    const auto outTiles = fillDividedTiles(nTilesOnDim, outputShape);

    for (const auto& outTile : outTiles) {
        const auto padsTileConf = backInferPadsTile(outTile, outputShape, VPU::toPadInfo(pads));
        const auto tilePad = VPU::getPaddingAttr(rewriter.getContext(), padsTileConf);

        origOp.addWorkload(rewriter, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad,
                           mpeByType(inElemType, outElemType, origOp));
    }
}

template <class ConcreteOp>
mlir::LogicalResult GenericNCERewrite<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto outputShape = getShape(origOp.output());
    auto pads = origOp.pad();

    const auto inElemType = origOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = origOp.output().getType().template cast<vpux::NDTypeInterface>().getElementType();

    rewriter.updateRootInPlace(origOp, [&]() {
        addDPUTasks(rewriter, origOp, pads, inElemType, outElemType, outputShape, _numDPU, _arch);
    });

    return mlir::success();
}

//
// EltwiseNCERewrite
//

class EltwiseNCERewrite final : public mlir::OpRewritePattern<VPU::NCEEltwiseOp> {
public:
    EltwiseNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::NCEEltwiseOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult EltwiseNCERewrite::matchAndRewrite(VPU::NCEEltwiseOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp.getContext();

    const auto outputShape = getShape(origOp.output());
    auto pads = VPU::PaddingAttr::get(getIntAttr(rewriter, 0), getIntAttr(rewriter, 0), getIntAttr(rewriter, 0),
                                      getIntAttr(rewriter, 0), ctx);

    const auto inElemType = origOp.input1().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();

    rewriter.updateRootInPlace(origOp, [&]() {
        addDPUTasks(rewriter, origOp, pads, inElemType, outElemType, outputShape, _numDPU, _arch);
    });

    return mlir::success();
}

//
// SplitNCEOpsOntoWorkloads
//

class SplitNCEOpsOntoWorkloadsPass final : public SplitNCEOpsOntoWorkloadsBase<SplitNCEOpsOntoWorkloadsPass> {
public:
    explicit SplitNCEOpsOntoWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SplitNCEOpsOntoWorkloadsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    VPUX_THROW_UNLESS(mpeMap.find(arch) != mpeMap.end(), "Failed to map MPE mode to target arch");

    auto nceCluster = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::DPU));
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GenericNCERewrite<VPU::NCEConvolutionOp>>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<GenericNCERewrite<VPU::NCEMaxPoolOp>>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<GenericNCERewrite<VPU::NCEDepthConvolutionOp>>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<EltwiseNCERewrite>(&ctx, dpuExec.count(), arch, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            return !nceOp.workloads().empty();
        }
        return true;
    });
    target.addLegalOp<VPU::DPUWorkloadOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitNCEOpsOntoWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(Logger log) {
    return std::make_unique<SplitNCEOpsOntoWorkloadsPass>(log);
}
