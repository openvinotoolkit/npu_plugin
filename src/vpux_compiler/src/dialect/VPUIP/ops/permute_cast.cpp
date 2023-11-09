//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::Value VPUIP::PermuteCastOp::getViewSource() {
    return source();
}

//
// fold
//

mlir::OpFoldResult vpux::VPUIP::PermuteCastOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reorder(DimsOrder::fromAffineMap(dst_order()));
    }

    return nullptr;
}

mlir::LogicalResult vpux::VPUIP::PermuteCastOp::verify() {
    const auto op = getOperation();
    auto distributedInType = source().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = result().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        auto inputDistribution = distributedInType.getDistribution();
        auto outputDistribution = distributedOutType.getDistribution();

        auto isMemoryFullSizeMode = [&](VPU::DistributionMode mode) -> bool {
            return VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED);
        };

        // TODO: Remove check skip once passes properly propagate distributed attrs through view ops
        // E#88457
        if (!isMemoryFullSizeMode(inputDistribution.getMode().getValue()) &&
            !isMemoryFullSizeMode(outputDistribution.getMode().getValue())) {
            auto expectedOutputDistribution = applyPermutationOnDistributedTensorAttr(
                    inputDistribution, mem_perm(), distributedInType.getDimsOrder(), distributedOutType.getDimsOrder());
            if (outputDistribution != expectedOutputDistribution) {
                return errorAt(op,
                               "PermuteCast input and output distributions are incompatible: in = {0}, out = {1},"
                               "expected = {2}",
                               inputDistribution, outputDistribution, expectedOutputDistribution);
            }
        }
    }

    const auto inType = source().getType().cast<vpux::NDTypeInterface>();
    const auto outType = result().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "PermuteCast input and output must have the same number of elements");
    }

    return mlir::success();
}
