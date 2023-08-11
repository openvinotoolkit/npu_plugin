//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

bool checkFactors(const Factors& factors, int64_t kernelSize = 0, bool onlyCheckLarger = false) {
    const auto hasZeroFactors = factors.larger == 0 || factors.smaller == 0;
    const auto factorLessThanKernelSize = factors.larger * factors.smaller < kernelSize;
    const auto hasUnsupportedFactors = onlyCheckLarger ? factors.larger > VPU::NCEInvariant::MAX_KERNEL_SIZE
                                                       : factors.larger > VPU::NCEInvariant::MAX_KERNEL_SIZE ||
                                                                 factors.smaller > VPU::NCEInvariant::MAX_KERNEL_SIZE;
    const auto hasBadFactors = factors.larger * factors.smaller > (kernelSize + factors.smaller / 2);
    return !(hasZeroFactors || factorLessThanKernelSize || hasUnsupportedFactors || hasBadFactors);
    // those last 2 checks have the main scope of finding the best suited factors:
    // if one of the last 2 checks fails it means that the gap between product of
    // those 2 factors and original kernel size is too big, which generates larger overlapping area
}

Factors getFactorsAround(int64_t kernelSize, int64_t pad) {
    const auto& candidateFactors = getFactorsList(kernelSize + pad);
    if (!candidateFactors.empty()) {
        return candidateFactors.back();
    }
    return {};
}

Factors getFactorsAroundWithLimit(int64_t kernelSize, int64_t pad, int64_t limit) {
    const auto& candidateFactors = getFactorsListWithLimitation(kernelSize + pad, limit);
    if (!candidateFactors.empty()) {
        return candidateFactors.back();
    }
    return {};
}

Optional<Factors> vpux::IE::getFactors(int64_t kernelSize, int64_t& padValue) {
    padValue = 1;
    const auto& allFactors = getFactorsList(kernelSize);
    if (!allFactors.empty() && checkFactors(allFactors.back(), kernelSize)) {
        return allFactors.back();
    }

    while (padValue < kernelSize) {
        const auto& factors = getFactorsAround(kernelSize, padValue);
        if (checkFactors(factors, kernelSize)) {
            return factors;
        }
        padValue++;
    }

    return None;
}

//
// For iteration splitting, "Larger" means the final size distribution after splitting
// In current iteration, the another factor is larger and it will enter next round to split into smaller factor
//
Optional<Factors> vpux::IE::getFactorsWithSupportedLarger(int64_t kernelSize, int64_t& padValue) {
    const auto limit = VPU::NCEInvariant::MAX_KERNEL_SIZE;
    const auto& allLimitFactors = getFactorsListWithLimitation(kernelSize, limit);
    if (!allLimitFactors.empty() && allLimitFactors.back().larger != 1 &&
        checkFactors(allLimitFactors.back(), kernelSize, true)) {
        return allLimitFactors.back();
    }

    while (padValue < kernelSize) {
        const auto& factors = getFactorsAroundWithLimit(kernelSize, padValue, limit);
        if (factors.larger != 1 && checkFactors(factors, kernelSize, true)) {
            return factors;
        }
        padValue++;
    }

    return None;
}

bool vpux::IE::hasSupportedKernels(ArrayRef<int64_t> kernelSize) {
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    return KY <= VPU::NCEInvariant::MAX_KERNEL_SIZE && KX <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
};

bool vpux::IE::isPoolingKernelSizeValid(int64_t kernelSize) {
    int64_t padValue = 1;
    return getFactors(kernelSize, padValue).hasValue();
}
