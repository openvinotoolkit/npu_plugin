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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEConv(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'", OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                       weightTableNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::AVEPOOL || op.task_type() == EMU::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", EMU::NCETaskType::AVEPOOL,
                      EMU::NCETaskType::MAXPOOL, op.task_type());

    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::ELTWISE, op.task_type());

    if (op.weight_table() != nullptr) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() != nullptr) {
        return errorAt(op, "kernel_size should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() != nullptr) {
        return errorAt(op, "kernel_strides should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() != nullptr) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", op.task_type());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEDWConv(EMU::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == EMU::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      EMU::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'", OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                       weightTableNumElements);
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::EMU::verifyOp(EMU::NCEClusterTaskOp op) {
    if (op.task_type() == EMU::NCETaskType::CONV) {
        if (mlir::failed(verifyNCEConv(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::MAXPOOL || op.task_type() == EMU::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == EMU::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(op))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", op.task_type());
    }

    for (auto& ppeOp : op.ppe().getOps()) {
        if (!mlir::isa<EMU::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    return mlir::success();
}
