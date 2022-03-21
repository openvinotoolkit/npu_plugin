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

#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

bool checkFactors(const Factors& factors, int64_t kernelSize = 0) {
    const auto hasZeroFactors = factors.larger == 0 || factors.smaller == 0;
    const auto factorLessThanKernelSize = factors.larger * factors.smaller < kernelSize;
    const auto hasUnsupportedFactors =
            factors.larger > VPU::NCEInvariant::MAX_KERNEL_SIZE || factors.smaller > VPU::NCEInvariant::MAX_KERNEL_SIZE;
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

bool factorsValid(int64_t kernelSize) {
    const auto& allFactors = getFactorsList(kernelSize);
    if (!allFactors.empty() && checkFactors(allFactors.back(), kernelSize)) {
        return true;
    }

    int64_t padValue = 1;
    while (padValue < kernelSize) {
        const auto& factors = getFactorsAround(kernelSize, padValue);
        if (checkFactors(factors, kernelSize)) {
            return true;
        }
        padValue++;
    }

    return false;
}

bool vpux::IE::hasSupportedKernels(ArrayRef<int64_t> kernelSize) {
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    return KY <= VPU::NCEInvariant::MAX_KERNEL_SIZE && KX <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
};

bool vpux::IE::isGlobalPoolingKernelSupported(mlir::Operation* op) {
    const auto inDataType = op->getOperand(0).getType().cast<NDTypeInterface>();
    const auto inDataShape = inDataType.getShape().raw();
    const llvm::SmallVector<int64_t> kernelSize = {inDataShape[Dims4D::Act::H.ind()],
                                                   inDataShape[Dims4D::Act::W.ind()]};
    if (IE::hasSupportedKernels(kernelSize)) {
        return true;
    }

    const llvm::SmallVector<int64_t> strides = {1, 1};
    const auto maxKernelSizeSupported =
            VPU::NCEInvariant::MAX_KERNEL_SIZE *
            VPU::NCEInvariant::MAX_KERNEL_SIZE;  // we can only get 2 factors and max kernel should be 11 * 11 = 121
    auto unsupportedKernelCheck = [&](int32_t kernelInd, int32_t actInd, int32_t strideInd) {
        return ((kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]) ||
                kernelSize[kernelInd] > maxKernelSizeSupported);
    };

    if (unsupportedKernelCheck(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind())) {
        return false;
    }
    if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
        return false;
    }

    return factorsValid(kernelSize[Dims4D::Kernel::X.ind()]) && factorsValid(kernelSize[Dims4D::Kernel::X.ind()]);
}
