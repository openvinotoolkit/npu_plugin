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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::DistributedCastOp::getViewSource() {
    return input();
}

//
// fold
//

mlir::OpFoldResult VPUIP::DistributedCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// verifyOp
//

mlir::LogicalResult VPUIP::verifyOp(VPUIP::DistributedCastOp op) {
    const auto inDistributedType = op.input().getType().cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = op.output().getType().cast<VPUIP::DistributedBufferType>();

    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        return errorAt(op->getLoc(), "Mismatch between buffer shapes for input ({0}) and output ({1}).",
                       inDistributedType.getShape(), outDistributedType.getShape());
    }

    if (inDistributedType.getElementType() != outDistributedType.getElementType()) {
        return errorAt(op->getLoc(), "Mismatch between buffer element types for input ({0}) and output ({1}).",
                       inDistributedType.getElementType(), outDistributedType.getElementType());
    }

    if (inDistributedType.getLayout() != outDistributedType.getLayout()) {
        return errorAt(op->getLoc(), "Mismatch between buffer layouts for input ({0}) and output ({1}).",
                       inDistributedType.getLayout(), outDistributedType.getLayout());
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        return errorAt(op->getLoc(), "Mismatch between buffer memspaces for input ({0}) and output ({1}).",
                       inDistributedType.getMemSpace(), outDistributedType.getMemSpace());
    }

    const auto inDistributionAttr = inDistributedType.getDistribution();
    const auto outDistributionAttr = outDistributedType.getDistribution();

    if (inDistributionAttr.num_clusters() != outDistributionAttr.num_clusters()) {
        return errorAt(op->getLoc(), "Mismatch between buffer number of clusters for input ({0}) and output ({1}).",
                       inDistributionAttr.num_clusters(), outDistributionAttr.num_clusters());
    }

    const auto inDistributionMode = inDistributionAttr.mode().getValue();
    const auto outDistributionMode = inDistributionAttr.mode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::areDistributionModesCompatible(inDistributionMode, outDistributionMode).failed()) {
            return errorAt(op->getLoc(), "Incompatible distribution modes for input ({0}) and output ({1}).",
                           VPU::stringifyDistributionMode(inDistributionMode),
                           VPU::stringifyDistributionMode(outDistributionMode));
        }
    }

    return mlir::success();
}
