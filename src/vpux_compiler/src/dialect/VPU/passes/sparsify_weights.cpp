//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/sparsity_strategy.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// SparsifyWeightsPass
//

class SparsifyWeightsPass final : public VPU::SparsifyWeightsBase<SparsifyWeightsPass> {
public:
    explicit SparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic, Optional<double> manualThreshold, Logger log)
            : _heuristic(heuristic), _manualThreshold(manualThreshold) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    VPU::WeightsSparsityHeuristic _heuristic;
    Optional<double> _manualThreshold;
};

//
// safeRunOnFunc
//

void SparsifyWeightsPass::safeRunOnFunc() {
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto module = getOperation();
    auto& ctx = getContext();

    const auto arch = VPU::getArch(module);

    std::unique_ptr<BaseWeightsSparsityStrategy> enablementStrategy;
    if (_heuristic == VPU::WeightsSparsityHeuristic::CMX) {
        _log.trace("Using CMX-based heuristic");
        const Byte availableCMX = VPU::getTotalCMXSize(module);
        enablementStrategy = std::make_unique<CMXConsumptionBasedWeightsSparsityStrategy>(
                availableCMX, CMX_BASED_STRATEGY_DEFAULT_INTERVALS, _manualThreshold);
    } else if (_heuristic == VPU::WeightsSparsityHeuristic::RATIO) {
        _log.trace("Using ratio-based heuristic");
        enablementStrategy = std::make_unique<RatioBasedWeightsSparsityStrategy>(
                WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD, WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD, _manualThreshold);
    } else {
        VPUX_THROW("Unsupported heuristic: {0}", _heuristic);
    }

    int64_t numCandidatesSparseWeights = 0;
    int64_t numSparsifiedWeights = 0;

    func->walk([&](VPU::SparseOpInterface sparsifiableOp) {
        if (!VPU::supportsSparseWeights(sparsifiableOp.getOperation())) {
            return;
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(sparsifiableOp.getOperation());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Unexpected non-NCE operation that supports weights sparsity");

        const auto weights = nceOp.getWeightsOperand();
        if (weights == nullptr) {
            return;
        }

        _log.trace("Op '{0}' at '{1}' is a candidate for sparsifying its weights", sparsifiableOp->getName(),
                   sparsifiableOp->getLoc());
        auto innerLog = _log.nest();

        ++numCandidatesSparseWeights;

        if (weights.getType().isa<VPU::SparseTensorType>()) {
            innerLog.trace("Weights are already sparse");
            return;
        }

        auto weightsOp = weights.getDefiningOp<Const::DeclareOp>();
        if (weightsOp == nullptr) {
            innerLog.trace("Expected weights parent to be constant, but got '{0}'", weights.getDefiningOp()->getName());
            return;
        }

        if (VPU::NCEInvariant::isCompressConvolution(arch, sparsifiableOp)) {
            innerLog.trace("Operation uses the compressed convolution feature. Skipping");
            return;
        }

        const auto inputType = sparsifiableOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto hasFloatInput = inputType.getElementType().isa<mlir::FloatType>();
        if (!enablementStrategy->shouldSparsifyWeights(innerLog, weights, hasFloatInput)) {
            innerLog.trace("Weights will not be sparsified", sparsifiableOp->getName(), sparsifiableOp->getLoc());
            return;
        }

        innerLog.trace("Sparsifying weights for op '{0}' at '{1}'", sparsifiableOp->getName(),
                       sparsifiableOp->getLoc());

        mlir::OpBuilder builder(weightsOp);

        auto weightsContent = weightsOp.contentAttr();
        auto sparsityMapContent = weightsContent.getSparsityMap();
        auto sparsifiedContent = weightsContent.sparsify(false);

        auto sparsifyTransformation = sparsifiedContent.getTransformations().back().dyn_cast<Const::SparsifyAttr>();
        VPUX_THROW_UNLESS(sparsifyTransformation != nullptr, "Missing Sparsify transformation");
        const auto numElemsAttr = sparsifyTransformation.getNumActualElements();
        const auto axisAttr = getIntAttr(&ctx, Dims4D::Filter::OC.ind());
        const auto alignmentAttr = getIntAttr(&ctx, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
        auto compressionSchemeAttr = VPU::CompressionSchemeAttr::get(&ctx, axisAttr, numElemsAttr, alignmentAttr);

        const auto sparsifiedWeights =
                builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsifiedContent.getType(), sparsifiedContent);
        const auto sparsityMap =
                builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsityMapContent.getType(), sparsityMapContent);
        const auto groupedView =
                builder.create<VPU::GroupSparseTensorOp>(weightsOp.getLoc(), sparsifiedWeights->getResult(0),
                                                         sparsityMap->getResult(0), true, compressionSchemeAttr);

        weightsOp->replaceAllUsesWith(groupedView);
        weightsOp->erase();

        ++numSparsifiedWeights;
    });

    _log.trace("Sparsified weights for {0} operations out of {1} candidates", numSparsifiedWeights,
               numCandidatesSparseWeights);
}

}  // namespace

//
// createSparsifyWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic,
                                                                 Optional<double> manualThreshold, Logger log) {
    return std::make_unique<SparsifyWeightsPass>(heuristic, manualThreshold, log);
}
