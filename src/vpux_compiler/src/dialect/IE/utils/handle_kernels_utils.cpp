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
    return true;
}
