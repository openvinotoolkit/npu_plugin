//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/cost_model.hpp"
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

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

namespace {

// up bound for workload numbers
constexpr size_t MAX_SPLIT_NUMBER = 50;

VPU::MPEMode getMpeModeForKmb(mlir::Type inElemType, mlir::Type outElemType, mlir::Operation*, ShapeRef shape) {
    if (inElemType.isa<mlir::quant::QuantizedType>() || outElemType.isa<mlir::quant::QuantizedType>()) {
        const double W = static_cast<double>(shape[Dims4D::Act::W]);
        const double H = static_cast<double>(shape[Dims4D::Act::H]);
        // VPU::MPEMode::MATRIX process tensor using W=4 H=4 parts, calculate grid cells count for it
        const double matrixPartsCount = std::ceil(W / 4.0) * std::ceil(H / 4.0);
        // VPU::MPEMode::VECTOR process tensor using W=16 H=1 parts, calculate grid cells count for it
        const double vectorPartsCount = std::ceil(W / 16.0) * H;
        // Cells count is in direct ratio with work size, so choose smaller one
        return (vectorPartsCount <= matrixPartsCount) ? VPU::MPEMode::VECTOR : VPU::MPEMode::MATRIX;
    }

    if (inElemType.isF16() || inElemType.isBF16() || outElemType.isF16() || outElemType.isBF16()) {
        return VPU::MPEMode::VECTOR_FP16;
    }

    // Let's fall back to vector (might be a bad idea though).
    return VPU::MPEMode::VECTOR;
}

VPU::MPEMode getMpeModeForMtl(mlir::Type, mlir::Type, mlir::Operation* operation, ShapeRef) {
    if (mlir::isa<VPU::NCEConvolutionOp>(operation)) {
        return VPU::MPEMode::CUBOID_16x16;
    } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(operation) || mlir::isa<VPU::NCEMaxPoolOp>(operation)) {
        return VPU::MPEMode::CUBOID_4x16;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(operation)) {
        return VPU::MPEMode::CUBOID_8x16;
    }

    return VPU::MPEMode::CUBOID_16x16;
}

using GetMpeModeCb = VPU::MPEMode (*)(mlir::Type, mlir::Type, mlir::Operation*, ShapeRef);

const EnumMap<VPU::ArchKind, GetMpeModeCb> mpeMap = {
        {VPU::ArchKind::KMB, getMpeModeForKmb},
        {VPU::ArchKind::TBH, getMpeModeForKmb},
        {VPU::ArchKind::MTL, getMpeModeForMtl},
};

// for workloads in sub tensors, offsets need to be from original full output tensor
void addSubTensorOffset(TileInfo& tileInfo, ShapeRef tensorOffset) {
    VPUX_THROW_WHEN(tileInfo.offsets.size() != tensorOffset.size(),
                    "Invalid size for TileInfo.offset {0} and sub tensor offset {1}", tileInfo.offsets.size(),
                    tensorOffset.size());
    for (auto d : irange(tileInfo.offsets.size())) {
        const auto dim = Dim(d);
        tileInfo.offsets[dim] += tensorOffset[dim];
    }
}

void addWorkload(mlir::PatternRewriter& rewriter, VPU::NCEOpInterface origOp,
                 const VPUIP::WorkloadCostParams& costParams, std::shared_ptr<VPUNN::VPUCostModel> costModel,
                 mlir::IntegerAttr clusterId = nullptr, ShapeRef subTensorOffset = {}) {
    VPUIP::DpuTiler dpuTiler(costParams.outputShape, costParams.mpeMode, costModel);
    dpuTiler.tileOverH(costParams.numDPU);

    const auto& splitNumPool = dpuTiler.generateSplitNumberPool(costParams.numDPU, MAX_SPLIT_NUMBER);
    for (const auto& splitNum : splitNumPool) {
        // Depthwise convolution doesn't support SplitOverHW
        if (mlir::isa<VPU::NCEDepthConvolutionOp>(origOp)) {
            dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_W);
        } else {
            dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_HW);
        }
        if (costParams.isTileOverZSupported) {
            dpuTiler.tileOverZ(splitNum);
        }
    }

    // select workload with minimum cost
    uint32_t minimumHardwareExecutionCost = std::numeric_limits<uint32_t>::max();
    const auto& splitCandidates = dpuTiler.getSplitPool();
    auto minCostIter = splitCandidates.begin();
    for (auto iter = splitCandidates.begin(); iter != splitCandidates.end(); iter++) {
        auto tiles = *iter;
        if (clusterId != nullptr) {
            for (auto& tile : tiles) {
                addSubTensorOffset(tile, subTensorOffset);
            }
        }
        auto hardwareExecutionCost = dpuTiler.cost(tiles, costParams);
        if (minimumHardwareExecutionCost > hardwareExecutionCost) {
            minimumHardwareExecutionCost = hardwareExecutionCost;
            minCostIter = iter;
        }
    }

    auto outTiles = *minCostIter;
    auto kernel = origOp.getKernelSize();
    auto strides = origOp.getStrides();
    for (auto& outTile : outTiles) {
        if (clusterId != nullptr) {
            addSubTensorOffset(outTile, subTensorOffset);
        }
        const auto padsTileConf =
                backInferPadsTile(outTile, costParams.fullInputShape, costParams.padInfo, kernel, strides);
        auto tilePad = VPU::getPaddingAttr(rewriter.getContext(), padsTileConf);
        origOp.addWorkload(rewriter, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad, costParams.mpeMode,
                           clusterId);
    }
}

//
// GenericNCERewrite
//

template <class ConcreteOp>
class GenericNCERewrite final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch,
                      std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log)
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
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    Logger _log;
};

void addDPUTasks(mlir::PatternRewriter& rewriter, VPU::NCEOpInterface origOp, VPUIP::WorkloadCostParams& costParams,
                 std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    if (auto clusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(origOp->getParentOp())) {
        const auto outputs = clusterOp->getResults();
        VPUX_THROW_UNLESS(outputs.size() == 1, "Wrong outputs size: {0}", outputs.size());

        const auto output = *outputs.begin();
        auto distributedOutputType = output.getType().dyn_cast<VPU::DistributedTensorType>();
        VPUX_THROW_WHEN(distributedOutputType == nullptr, "Wrong output type {0} for NCEClusterTilingOp",
                        output.getType());

        const auto outputSubTensorShapes = distributedOutputType.getPerClusterComputeShapes();
        const auto outputSubTensorOffsets = distributedOutputType.getPerClusterComputeShapeOffsets();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != outputSubTensorOffsets.size(),
                        "sub tensor size:{0} not equal to offset size:{1}", outputSubTensorShapes.size(),
                        outputSubTensorOffsets.size());

        const auto inputs = clusterOp->getOperands();
        VPUX_THROW_UNLESS(inputs.size() >= 1, "Wrong inputs size: {0}", inputs.size());

        const auto input = *inputs.begin();
        auto distributedInputType = input.getType().dyn_cast<VPU::DistributedTensorType>();
        VPUX_THROW_WHEN(distributedInputType == nullptr, "Wrong input type {0} for NCEClusterTilingOp",
                        input.getType());

        const auto inputSubTensorShapes = distributedInputType.getPerClusterComputeShapes();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != inputSubTensorShapes.size(),
                        "output tensor size:{0} not equal to input tensor size:{1}", outputSubTensorShapes.size(),
                        inputSubTensorShapes.size());

        for (size_t clusterId = 0; clusterId < outputSubTensorShapes.size(); clusterId++) {
            auto clusterIdAttr = getIntAttr(origOp->getContext(), clusterId);
            costParams.inputShape = inputSubTensorShapes[clusterId];
            costParams.outputShape = outputSubTensorShapes[clusterId];

            addWorkload(rewriter, origOp, costParams, costModel, clusterIdAttr, outputSubTensorOffsets[clusterId]);
        }
    } else {
        addWorkload(rewriter, origOp, costParams, costModel);
    }
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
mlir::LogicalResult GenericNCERewrite<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    auto input = getOpInput(origOp);
    auto inElemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = origOp.output().getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto inputShape = getShape(input);
    auto outputShape = getShape(origOp.output());
    auto pads = nceOp.getPad();

    const auto mpeByType = mpeMap.at(_arch);
    const auto mpeMode = mpeByType(inElemType, outElemType, origOp, outputShape);

    VPUIP::WorkloadCostParams params;
    params.mpeMode = mpeMode;
    params.dataType = inElemType;
    params.numDPU = _numDPU;
    params.arch = _arch;
    params.fullInputShape = inputShape.raw();
    params.inputShape = inputShape.raw();
    params.outputShape = outputShape.raw();
    params.padInfo = VPU::toPadInfo(pads);
    params.kernelSize = nceOp.getKernelSize();
    params.kernelStride = nceOp.getStrides();
    params.isTileOverZSupported = params.mpeMode == VPU::MPEMode::VECTOR;

    llvm::TypeSwitch<mlir::Operation*, void>(origOp.getOperation())
            .template Case<VPU::NCEConvolutionOp>([&params](VPU::NCEConvolutionOp origOp) {
                const auto inOrder = DimsOrder::fromValue(origOp.input());
                const auto isCMajor = inOrder == DimsOrder::NCHW;
                params.nceTaskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
                params.isTileOverZSupported |= !isCMajor;
            })
            .template Case<VPU::NCEDepthConvolutionOp>([&params](VPU::NCEDepthConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::DWCONV;
            })
            .template Case<VPU::NCEMaxPoolOp>([&params](VPU::NCEMaxPoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::MAXPOOL;
            })
            .template Case<VPU::NCEEltwiseOp>([&params](VPU::NCEEltwiseOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
                params.isTileOverZSupported = false;
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

    const auto costModel = VPU::createCostModel(arch);

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
