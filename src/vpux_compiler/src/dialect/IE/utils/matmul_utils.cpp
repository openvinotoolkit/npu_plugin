//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/utils/matmul_utils.hpp"

namespace vpux {
namespace IE {

size_t getShapeSize(const vpux::NDTypeInterface type) {
    auto shape = type.getShape();
    size_t typeSize = 0;
    for (auto dim : shape) {
        if (dim != 1) {
            typeSize++;
        }
    }
    return typeSize;
}

SmallVector<mlir::Operation*> getMatMulParentOps(IE::MatMulOp origOp) {
    // return the required parent ops for MatMul
    // the returned type/order is {actParent, filterParent}
    SmallVector<mlir::Operation*> parentList;
    auto isOpIgnorable = [](mlir::Operation* testOp) {
        return mlir::isa<IE::ReshapeOp>(testOp);
    };
    for (auto opIndex = 0; opIndex < 2; opIndex++) {
        auto inputOp = origOp->getOperand(opIndex).getDefiningOp();
        while (inputOp != nullptr && isOpIgnorable(inputOp)) {
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }
        if (inputOp != nullptr) {
            parentList.push_back(inputOp);
        }
    }

    VPUX_THROW_UNLESS(parentList.size() == 2, "MatMul {0} does not have two parent ops", origOp->getLoc());

    mlir::Operation* actInputOp;
    mlir::Operation* weightInputOp;

    // consider the input with rank=2 to be the activation
    // and rank=1 to be the weight
    for (auto parentOp : parentList) {
        if (getShapeSize(parentOp->getResult(0).getType().cast<vpux::NDTypeInterface>()) == size_t(1)) {
            weightInputOp = parentOp;
        } else if (getShapeSize(parentOp->getResult(0).getType().cast<vpux::NDTypeInterface>()) > size_t(1)) {
            actInputOp = parentOp;
        } else {
            return SmallVector<mlir::Operation*>{nullptr, nullptr};
        }
    }

    return SmallVector<mlir::Operation*>{actInputOp, weightInputOp};
}

bool oneDimMatch(vpux::NDTypeInterface actType, vpux::NDTypeInterface filterType) {
    // The size of filter must match the channel size of activation
    auto filterSize = 1;
    for (auto filterShapeDim : filterType.getShape()) {
        if (filterShapeDim != 1) {
            filterSize = filterShapeDim;
            break;
        }
    }
    return actType.getShape()[Dims4D::Act::C] == filterSize;
}

bool isKernelDivisible(ShapeRef weightShape) {
    int64_t totalShape = 1;
    for (auto dim : weightShape) {
        if (dim != 1) {
            totalShape = dim;
        }
    }
    auto kernelFactors = getKernelFactors(totalShape);
    if (kernelFactors.size() != 2)
        return false;
    return std::all_of(kernelFactors.begin(), kernelFactors.end(), [](int64_t factor) {
        return factor >= 1 || factor <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
    });
}

bool checkPermuteMatMulPattern(IE::MatMulOp origOp) {
    auto parentOps = getMatMulParentOps(origOp);
    mlir::Operation* actInputOp = parentOps[0];
    mlir::Operation* weightInputOp = parentOps[1];

    if (actInputOp == nullptr || weightInputOp == nullptr) {
        return false;
    }

    if (!oneDimMatch(actInputOp->getResult(0).getType().cast<vpux::NDTypeInterface>(),
                     weightInputOp->getResult(0).getType().cast<vpux::NDTypeInterface>())) {
        return false;
    }

    // match the Transpose
    if (!mlir::isa<IE::TransposeOp>(actInputOp)) {
        return false;
    }

    return isKernelDivisible(weightInputOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape());
}

SmallVector<int64_t, 2> getKernelFactors(int64_t total) {
    int64_t kx = sqrt(total);
    int64_t ky = total / kx;
    while (total % kx != 0) {
        kx--;
        ky = total / kx;
    }
    std::cout << "divide " << total << " into " << kx << " " << ky << std::endl;
    return {kx, ky};
}

}  // namespace IE
}  // namespace vpux
