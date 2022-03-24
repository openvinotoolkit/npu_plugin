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

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// fold
//

mlir::OpFoldResult VPU::DistributedCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// verifyOp
//

mlir::LogicalResult VPU::verifyOp(VPU::DistributedCastOp op) {
    const auto logCb = [op](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    const auto inDistributedType = op.input().getType().cast<VPU::DistributedTensorType>();
    const auto outDistributedType = op.output().getType().cast<VPU::DistributedTensorType>();

    return isDistributedCastCompatible(inDistributedType, outDistributedType, logCb);
}
