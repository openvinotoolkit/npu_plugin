//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/sparsity_strategy.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <limits>

using namespace vpux;
namespace vpux {
namespace VPU {

namespace NCESparsity {

using namespace VPU::NCESparsity;

bool BaseWeightsSparsityStrategy::shouldSparsifyWeights(Logger&, mlir::Value, bool) {
    return false;
}

RatioBasedWeightsSparsityStrategy::RatioBasedWeightsSparsityStrategy(double floatRatioThreshold,
                                                                     double intRatioThreshold,
                                                                     Optional<double> manualThreshold)
        : _floatRatioThreshold(floatRatioThreshold),
          _intRatioThreshold(intRatioThreshold),
          _manualThreshold(manualThreshold) {
}

bool RatioBasedWeightsSparsityStrategy::shouldSparsifyWeights(Logger& log, mlir::Value sparsifyCandidateOperand,
                                                              bool hasFloatInput) {
    VPUX_THROW_UNLESS(isSparsifiableWeightsOperand(sparsifyCandidateOperand),
                      "shouldSparsifyWeights can handle only sparsifiable values, but got '{0}'",
                      sparsifyCandidateOperand);
    const auto weightsOp = sparsifyCandidateOperand.getDefiningOp<Const::DeclareOp>();

    const auto actualSparsityRatio = getSparsityRatio(weightsOp);
    auto sparsityRatioThreshold = hasFloatInput ? _floatRatioThreshold : _intRatioThreshold;
    if (_manualThreshold.hasValue()) {
        sparsityRatioThreshold = _manualThreshold.getValue();
    }

    log.trace("Sparsity ratio {0}, threshold {1}", actualSparsityRatio, sparsityRatioThreshold);
    return std::isgreaterequal(actualSparsityRatio, sparsityRatioThreshold);
}

CMXBasedSparsityThreshold::CMXBasedSparsityThreshold(double cmxSizeRatio, double floatRatioThreshold,
                                                     double intRatioThreshold)
        : _cmxSizeRatio(cmxSizeRatio),
          _floatRatioThreshold(floatRatioThreshold),
          _intRatioThreshold(intRatioThreshold) {
}

CMXConsumptionBasedWeightsSparsityStrategy::CMXConsumptionBasedWeightsSparsityStrategy(
        Byte cmxMemorySize, const std::initializer_list<CMXBasedSparsityThreshold>& intervals,
        Optional<double> manualThreshold)
        : _cmxMemorySize(cmxMemorySize), _manualThreshold(manualThreshold) {
    for (const auto& interval : intervals) {
        const auto cmxRatio = interval._cmxSizeRatio;
        VPUX_THROW_WHEN(cmxRatio < 0. || cmxRatio >= 1.,
                        "CMX consumption ratio should be in (0;1) interval, but got {0}", cmxRatio);
        _intervals.emplace(cmxRatio, std::make_pair(interval._intRatioThreshold, interval._floatRatioThreshold));
    }
    const auto disabledSparsityRatioInterval = std::make_pair(CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO,
                                                              CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO);
    // Disable sparsity for 0.-first interval
    _intervals.emplace(0., disabledSparsityRatioInterval);
    // Disable sparsity for >1. interval
    _intervals.emplace(1., disabledSparsityRatioInterval);
}

bool CMXConsumptionBasedWeightsSparsityStrategy::shouldSparsifyWeights(Logger& log,
                                                                       mlir::Value sparsifyCandidateOperand,
                                                                       bool hasFloatInput) {
    VPUX_THROW_UNLESS(isSparsifiableWeightsOperand(sparsifyCandidateOperand),
                      "shouldSparsifyWeights can handle only sparsifiable values, but got '{0}'",
                      sparsifyCandidateOperand);
    const auto weightsOp = sparsifyCandidateOperand.getDefiningOp<Const::DeclareOp>();

    double sparsityRatioThreshold = 0.0;
    if (_manualThreshold.hasValue()) {
        sparsityRatioThreshold = _manualThreshold.getValue();
    } else {
        const auto weightsType = sparsifyCandidateOperand.getType().cast<vpux::NDTypeInterface>();
        const auto weightsSize = weightsType.getTotalAllocSize();
        const double cmxConsumptionRatio =
                checked_cast<double>(weightsSize.count()) / checked_cast<double>(_cmxMemorySize.count());
        log.trace("CMX consumption ratio {0}", cmxConsumptionRatio);
        auto intervalIt = _intervals.lower_bound(cmxConsumptionRatio);
        intervalIt--;
        const auto interval = intervalIt->second;
        sparsityRatioThreshold = hasFloatInput ? interval.second : interval.first;
    }

    const auto actualSparsityRatio = getSparsityRatio(weightsOp);
    log.trace("Sparsity ratio {0}, threshold {1}", actualSparsityRatio, sparsityRatioThreshold);
    return std::isgreaterequal(actualSparsityRatio, sparsityRatioThreshold);
}

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
