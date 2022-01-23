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
        return VPU::MPEMode::MATRIX;
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
                 VPU::ArchKind arch, bool isZTilingSupported) {
    const auto mpeByType = mpeMap.at(arch);
    auto mpeMode = mpeByType(inElemType, outElemType, origOp);

    VPUIP::DpuTiler dpuTiler(outputShape, {mpeMode});
    dpuTiler.tileOverH(numDPU);
#ifdef __linux__
    dpuTiler.generateSplitNumberPool(numDPU, 5);
    auto splitNumPool = dpuTiler.getSplitNumberPool();
    /*Eltwise ops (including HwConvert) are performed by the PPE, which does not support Z-Tiling*/
#if 0
    auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(origOp);
    auto nceType = nceOp->task_type();
    bool isZTilingSupported = true;
    switch (nceType) {
    case VPUIP::NCETaskType::ELTWISE: {
        isZTilingSupported = false;
        break;
    }
    case VPUIP::NCETaskType::CMCONV:
    case VPUIP::NCETaskType::DWCONV:
    case VPUIP::NCETaskType::MAXPOOL:
    case VPUIP::NCETaskType::AVEPOOL: {
        if (mpeMode != VPU::MPEMode::VECTOR) {
            isZTilingSupported = false;
        }
        break;
    }
    default:
        break;
    }
#endif
    if (isZTilingSupported) {
        for (auto& splitNum : splitNumPool) {
            dpuTiler.tileOverZ(splitNum);
        }
    }

    // select workload with minimum cost
    uint32_t bestScore = UINT32_MAX;
    int best = -1;
    const auto& splitCandidates = dpuTiler.getSplitPool();
    for (size_t idx = 0; idx < splitCandidates.size(); idx++) {
        auto score = dpuTiler.cost(nceOp, splitCandidates[idx], numDPU, arch);
        if (bestScore > score) {
            bestScore = score;
            best = idx;
        }
    }
    if (best == -1) {
        VPUX_THROW("no workload splits found!");
    }
    const auto& outTiles = splitCandidates[best];
#else
    VPUX_UNUSED(archKind);
    const auto& splitCandidates = dpuTiler.getSplitPool();
    VPUX_THROW_WHEN(splitCandidates.empty(), "No available workload dpu tiles found");
    const auto& outTiles = splitCandidates.front();
#endif

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

    const auto inElemType = origOp.input().getType().template cast<mlir::RankedTensorType>().getElementType();
    const auto outElemType = origOp.output().getType().template cast<mlir::RankedTensorType>().getElementType();

    const auto filterShape = origOp.rawFilterShapeAttr() != nullptr
                             ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()))
                             : getShape(origOp.filter());

    const auto mpeByType = mpeMap.at(_arch);
    const auto mpeMode = mpeByType(inElemType, outElemType, origOp);
    const auto isZTilingSupported = VPU::isZTilingSupported(origOp->getOperation(),mpeMode);
    rewriter.updateRootInPlace(origOp, [&]() {
        addDPUTasks(rewriter, origOp, pads, inElemType, outElemType, outputShape, _numDPU, _arch, isZTilingSupported);
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

    const auto inElemType = origOp.input1().getType().cast<mlir::RankedTensorType>().getElementType();
    const auto outElemType = origOp.output().getType().cast<mlir::RankedTensorType>().getElementType();
    const auto mpeByType = mpeMap.at(_arch);
    const auto mpeMode = mpeByType(inElemType, outElemType, origOp);
    const auto isZTilingSupported = VPU::isZTilingSupported(origOp->getOperation(),mpeMode);

    rewriter.updateRootInPlace(origOp, [&]() {
        addDPUTasks(rewriter, origOp, pads, inElemType, outElemType, outputShape, _numDPU, _arch, isZTilingSupported);
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
