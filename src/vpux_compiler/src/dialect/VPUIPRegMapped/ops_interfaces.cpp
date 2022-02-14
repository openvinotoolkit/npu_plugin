//
// Copyright (C) 2022 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/effects.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// TaskOpInterface
//

void vpux::VPUIPRegMapped::getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    VPUX_UNUSED(op);
    VPUX_UNUSED(effects);

    // TODO:: do VPUIPRegMapped ops have modelable effects?

    return;
}

mlir::SymbolRefAttr vpux::VPUIPRegMapped::getDMAEngine(uint32_t& numUnits, mlir::MLIRContext* ctx,
                                                       VPUIPRegMapped::DMAEngine engine) {
    VPUX_UNUSED(engine);

    numUnits = 1;
    return mlir::SymbolRefAttr::get(ctx, "DMAEngine_str");
}

mlir::SymbolRefAttr vpux::VPUIPRegMapped::getTaskOpExecutor(mlir::Operation* op, uint32_t& numUnits) {
    auto task = mlir::cast<VPUIPRegMapped::TaskOpInterface>(op);
    const auto taskType = task.getTaskType();

    switch (taskType) {
    case VPUIPRegMapped::TaskType::NNDMA:
        return VPUIPRegMapped::getDMAEngine(numUnits, op->getContext(), VPUIPRegMapped::DMAEngine::DMA_NN);
    default:
        VPUX_THROW("Unsupported task type '{0}'", taskType);
    }
}

//
// Legacy4D
//

mlir::LogicalResult vpux::VPUIPRegMapped::verifyLegacy4D(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());
        const auto order = DimsOrder::fromValue(val.get());

        if (shape.size() != 3 && shape.size() != 4) {
            return errorAt(op, "Got unsupported shape '{0}', only 3D/4D are supported", shape);
        }

        if (shape.size() == 3) {
            if (order != DimsOrder::CHW && order != DimsOrder::HWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only CHW and HWC are supported", order);
            }
        } else if (shape.size() == 4) {
            if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported", order);
            }

            if (shape.front() != 1) {
                return errorAt(op, "Batch size != 1 is not supported");
            }
        }
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops_interfaces.cpp.inc>
