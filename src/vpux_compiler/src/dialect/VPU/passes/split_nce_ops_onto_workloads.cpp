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

#include <file_utils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
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

SmallString getVPUNNModelFile(VPU::ArchKind archKind) {
    SmallString modelDir;
    // probe for OpenVINO runtime dir
    auto ovBuildDir = InferenceEngine::getIELibraryPath();

    modelDir = ovBuildDir;
    llvm::sys::path::append(modelDir, "vpunn");
    switch (archKind) {
    case VPU::ArchKind::KMB:
    case VPU::ArchKind::TBH:
        llvm::sys::path::append(modelDir, "vpu_2_0.vpunn");
        break;
    case VPU::ArchKind::MTL:
        llvm::sys::path::append(modelDir, "vpu_2_7.vpunn");
        break;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
    VPUX_THROW_UNLESS(llvm::sys::fs::exists(modelDir), "vpunn model {0} does not exist", modelDir);
    return modelDir;
}

//
// GenericNCERewrite
//

template <class ConcreteOp>
class GenericNCERewrite final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, VPUNN::VPUCostModel costModel,
                      Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx),
              _numDPU(numDPU),
              _arch(arch),
              _costModel(std::move(costModel)),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    VPUNN::VPUCostModel _costModel;
    Logger _log;
};

void addDPUTasks(mlir::PatternRewriter& rewriter, VPU::NCEOpInterface origOp,
                 const VPUIP::WorkloadCostParams& costParams, const VPUNN::VPUCostModel& costModel) {
    VPUIP::DpuTiler dpuTiler(costParams.outputShape, costParams.mpeMode, costModel);
    dpuTiler.tileOverH(costParams.numDPU);

    // This value is aligned with the max DPU task value in NCEClusterTask Op
    static const size_t MAX_SPLIT_NUMBER = 5;
    const auto& splitNumPool = dpuTiler.generateSplitNumberPool(costParams.numDPU, MAX_SPLIT_NUMBER);
    if (costParams.isZTilingSupported) {
        for (const auto& splitNum : splitNumPool) {
            dpuTiler.tileOverZ(splitNum);
        }
    }

    // select workload with minimum cost
    uint32_t minimumHardwareExecutionCost = std::numeric_limits<uint32_t>::max();
    size_t minCostIdx = 0;

    const auto& splitCandidates = dpuTiler.getSplitPool();
    for (size_t idx = 0; idx < splitCandidates.size(); idx++) {
        auto hardwareExecutionCost = dpuTiler.cost(splitCandidates[idx], costParams);
        if (minimumHardwareExecutionCost > hardwareExecutionCost) {
            minimumHardwareExecutionCost = hardwareExecutionCost;
            minCostIdx = idx;
        }
    }

    const auto& outTiles = splitCandidates[minCostIdx];

    for (const auto& outTile : outTiles) {
        const auto padsTileConf = backInferPadsTile(outTile, costParams.outputShape, costParams.padInfo);
        const auto tilePad = VPU::getPaddingAttr(rewriter.getContext(), padsTileConf);

        origOp.addWorkload(rewriter, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad, costParams.mpeMode);
    }
}

template <class ConcreteOp>
VPU::PaddingAttr getOpPadding(ConcreteOp origOp) {
    return origOp.pad();
}

template <>
VPU::PaddingAttr getOpPadding(VPU::NCEEltwiseOp origOp) {
    auto zeroAttr = getIntAttr(origOp->getContext(), 0);
    return VPU::PaddingAttr::get(zeroAttr, zeroAttr, zeroAttr, zeroAttr, origOp->getContext());
}

template <class ConcreteOp>
mlir::Value getOpInput(ConcreteOp origOp) {
    return origOp.input();
}

template <>
mlir::Value getOpInput(NCEEltwiseOp origOp) {
    return origOp.input1();
}

template <class ConcreteOp>
SmallVector<int64_t> getOpKernelSize(ConcreteOp origOp) {
    const auto filterShape = origOp.rawFilterShapeAttr() != nullptr
                                     ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()))
                                     : getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    return {KY, KX};
}

template <>
SmallVector<int64_t> getOpKernelSize(VPU::NCEMaxPoolOp origOp) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];
    return {KY, KX};
}

template <>
SmallVector<int64_t> getOpKernelSize(VPU::NCEEltwiseOp origOp) {
    VPUX_UNUSED(origOp);
    return {1, 1};
}

template <class ConcreteOp>
SmallVector<int64_t> getOpKernelStride(ConcreteOp origOp) {
    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];
    return {SY, SX};
}

template <>
SmallVector<int64_t> getOpKernelStride(VPU::NCEEltwiseOp origOp) {
    VPUX_UNUSED(origOp);
    return {1, 1};
}

template <class ConcreteOp>
mlir::LogicalResult GenericNCERewrite<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto input = getOpInput(origOp);
    auto inElemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = origOp.output().getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto inputShape = getShape(input);
    auto outputShape = getShape(origOp.output());
    auto pads = getOpPadding(origOp);

    const auto mpeByType = mpeMap.at(_arch);
    const auto mpeMode = mpeByType(inElemType, outElemType, origOp);

    VPUIP::WorkloadCostParams params;
    params.mpeMode = mpeMode;
    params.dataType = inElemType;
    params.numDPU = _numDPU;
    params.arch = _arch;
    params.inputShape = inputShape.raw();
    params.outputShape = outputShape.raw();
    params.padInfo = VPU::toPadInfo(pads);
    params.kernelSize = getOpKernelSize(origOp);
    params.kernelStride = getOpKernelStride(origOp);

    llvm::TypeSwitch<mlir::Operation*, void>(origOp.getOperation())
            .template Case<VPU::NCEConvolutionOp>([&params](VPU::NCEConvolutionOp origOp) {
                const auto inOrder = DimsOrder::fromValue(origOp.input());
                const auto isCMajor = inOrder == DimsOrder::NCHW;
                params.nceTaskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
                params.isZTilingSupported = !isCMajor || params.mpeMode == VPU::MPEMode::VECTOR;
            })
            .template Case<VPU::NCEDepthConvolutionOp>([&params](VPU::NCEDepthConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::DWCONV;
                params.isZTilingSupported = params.mpeMode == VPU::MPEMode::VECTOR;
                return params;
            })
            .template Case<VPU::NCEMaxPoolOp>([&params](VPU::NCEMaxPoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::MAXPOOL;
                params.isZTilingSupported = params.mpeMode == VPU::MPEMode::VECTOR;
            })
            .template Case<VPU::NCEEltwiseOp>([&params](VPU::NCEEltwiseOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
                params.isZTilingSupported = false;
            })
            .Default([](mlir::Operation*) {
                VPUX_THROW("Unsupported NCE type");
            });

    rewriter.updateRootInPlace(origOp, [&]() {
        addDPUTasks(rewriter, origOp, params, _costModel);
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
    auto vpunnModelFile = getVPUNNModelFile(arch);
    VPUNN::VPUCostModel costModel(vpunnModelFile.str().str());
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GenericNCERewrite<VPU::NCEConvolutionOp>>(&ctx, dpuExec.count(), arch, costModel, _log);
    patterns.insert<GenericNCERewrite<VPU::NCEMaxPoolOp>>(&ctx, dpuExec.count(), arch, costModel, _log);
    patterns.insert<GenericNCERewrite<VPU::NCEDepthConvolutionOp>>(&ctx, dpuExec.count(), arch, costModel, _log);
    patterns.insert<GenericNCERewrite<VPU::NCEEltwiseOp>>(&ctx, dpuExec.count(), arch, costModel, _log);

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
