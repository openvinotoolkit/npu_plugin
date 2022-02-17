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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIPRegMapped::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value weights,
        mlir::Value weight_table, mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_output,
        mlir::Value output_buff, vpux::VPUIPRegMapped::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued) {
    build(builder, state, output_buff.getType(), input, weights, weight_table, activation_window, parent_input,
          parent_output, output_buff, mlir::ValueRange{}, mlir::ValueRange{},
          vpux::VPUIPRegMapped::NCETaskTypeAttr::get(builder.getContext(), task_type), kernel_size, kernel_strides,
          kernel_padding, activation_window_channel_length, is_continued);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIPRegMapped::DPUTaskOp vpux::VPUIPRegMapped::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder,
                                                                             mlir::ArrayAttr start, mlir::ArrayAttr end,
                                                                             VPUIPRegMapped::PaddingAttr pad,
                                                                             VPUIPRegMapped::MPEMode mpeMode) {
    if (variants().empty()) {
        variants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&variants().front());

    return builder.create<VPUIPRegMapped::DPUTaskOp>(getLoc(), start, end, pad, mpeMode);
}

//
// NCEClusterTaskOp::inferLayoutInfo
//

void vpux::VPUIPRegMapped::NCEClusterTaskOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    llvm::TypeSwitch<mlir::Operation*, void>(origOp)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp) {
                info.setInput(0, DimsOrder::NHWC);
                info.setInput(1, DimsOrder::OYXI);
                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp) {
                info.setInput(0, DimsOrder::NHWC);
                info.setInput(1, DimsOrder::OYXI);
                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AddOp>([&](IE::AddOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AndOp>([&](IE::AndOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Default([](mlir::Operation* unknownOp) -> bool {
                VPUX_THROW("Operation '{0}' the operation is not supported by the DPU", unknownOp->getName());
            });
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEEltwise(VPUIPRegMapped::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIPRegMapped::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIPRegMapped::NCETaskType::ELTWISE, op.task_type());

    if (op.weight_table() != nullptr) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() != nullptr) {
        return errorAt(op, "activation_window should be empty for NCETaskType : '{0}'", op.task_type());
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

}  // namespace

mlir::LogicalResult vpux::VPUIPRegMapped::verifyOp(VPUIPRegMapped::DPUTaskOp op) {
    static const size_t NUM_WORKLOAD_DIMS = 3;

    if (op.start().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.start().size());
    }
    if (op.end().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.end().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIPRegMapped::verifyOp(VPUIPRegMapped::NCEClusterTaskOp op) {
    if (op.task_type() == VPUIPRegMapped::NCETaskType::CONV) {
    } else if (op.task_type() == VPUIPRegMapped::NCETaskType::MAXPOOL ||
               op.task_type() == VPUIPRegMapped::NCETaskType::AVEPOOL) {
    } else if (op.task_type() == VPUIPRegMapped::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIPRegMapped::NCETaskType::DWCONV) {
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", op.task_type());
    }

    size_t numDPUTasks = 0;
    for (auto& dpuOp : op.variants().getOps()) {
        if (!mlir::isa<VPUIPRegMapped::DPUTaskOp>(dpuOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'variants' region", dpuOp.getName());
        }

        ++numDPUTasks;
    }

    static const size_t MAX_NUM_DPUS_PER_CLUSTER = 5;
    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks > MAX_NUM_DPUS_PER_CLUSTER || numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be a total of {0}-{1} DPU Tasks per NCEClusterTask, but got {2}",
                       MIN_NUM_DPUS_PER_CLUSTER, MAX_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : op.ppe().getOps()) {
        if (!mlir::isa<VPUIPRegMapped::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    return mlir::success();
}

//
// NCEClusterTaskOp::serialize
//

namespace {}  // namespace

void vpux::VPUIPRegMapped::NCEClusterTaskOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
}
size_t vpux::VPUIPRegMapped::NCEClusterTaskOp::getBinarySize() {
    return 0;
}
