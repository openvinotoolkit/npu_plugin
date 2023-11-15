//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Visitors.h>
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

using namespace vpux;

namespace {
template <class T, typename = require_t<std::is_integral<T>>>
mlir::IntegerAttr getOptionalInt(mlir::MLIRContext* ctx, llvm::Optional<T> value) {
    return value.hasValue() ? getIntAttr(ctx, value.getValue()) : nullptr;
}

};  // namespace

VPUIP::DpuProfilingMetadataAttr vpux::getDpuProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned bufferId,
                                                              unsigned taskId, unsigned maxVariants,
                                                              llvm::Optional<unsigned> numVariants,
                                                              llvm::Optional<unsigned> clusterId) {
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, getIntAttr(ctx, bufferId), getIntAttr(ctx, taskId),
                                                getIntAttr(ctx, maxVariants), getOptionalInt(ctx, numVariants),
                                                getOptionalInt(ctx, clusterId));
}

std::string convertExecTypeToName(profiling::ExecutorType execType) {
    using profiling::ExecutorType;
    switch (execType) {
    case ExecutorType::ACTSHAVE:
        return "actshave";
    case ExecutorType::DMA_HW:
        return "dmahw";
    case ExecutorType::DMA_SW:
        return "dma";
    case ExecutorType::DPU:
        return "dpu";
    case ExecutorType::UPA:
        return "upa";
    case ExecutorType::WORKPOINT:
        return "pll";
    case ExecutorType::NONE:
    default:
        VPUX_THROW("Unknown execType");
    };
};

;  // namespace

profiling::ExecutorType vpux::convertDataInfoNameToExecType(StringRef name) {
    using profiling::ExecutorType;
    if (name == "actshave") {
        return ExecutorType::ACTSHAVE;
    } else if (name == "dmahw") {
        return ExecutorType::DMA_HW;
    } else if (name == "dma") {
        return ExecutorType::DMA_SW;
    } else if (name == "dpu") {
        return ExecutorType::DPU;
    } else if (name == "none") {
        return ExecutorType::NONE;
    } else if (name == "upa") {
        return ExecutorType::UPA;
    } else if (name == "pll") {
        return ExecutorType::WORKPOINT;
    }
    VPUX_THROW("Can not convert '{0}' to profiling::ExecutorType", name);
}

mlir::BlockArgument vpux::addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc,
                                                IE::CNNNetworkOp& netOp, mlir::MemRefType outputType,
                                                profiling::ExecutorType execType) {
    const auto name = convertExecTypeToName(execType);
    //
    // Declare and create additional output from network
    //
    auto funcType = netFunc.getFunctionType();
    auto newResultTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), makeArrayRef(outputType)));
    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(outputType)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);

    // If you hit this, IR have CNNNetworkOp without profilingOutputInfo region
    VPUX_THROW_WHEN(netOp.getProfilingOutputsInfo().empty(), "Could not add profiling output: no region added");

    const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();

    // Adding output to the user info
    auto outputUserResult =
            getTensorType(ndOutputType.getShape(), ndOutputType.getElementType(), ndOutputType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.getProfilingOutputsInfo().front().front());
    userInfoBuilder.create<IE::DataInfoOp>(mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profilingDataOutputInfo")),
                                           mlir::StringAttr::get(ctx, name), mlir::TypeAttr::get(outputUserResult),
                                           /*profilingSectionsCount=*/0);

    const mlir::Location suffixLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profiling_" + name));
    const auto argLoc = mlir::FusedLoc::get(ctx, {netFunc.getLoc(), suffixLoc});

    return netFunc.getBody().front().addArgument(outputType, argLoc);
}

bool vpux::isProfiledDmaTask(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();

    VPUX_THROW_WHEN(mlir::isa<VPUIP::NCEClusterTilingOp>(wrappedTaskOp),
                    "NCEClusterTiling is not expected at this stage of compilation");

    return mlir::isa_and_nonnull<VPUIP::DMATypeOpInterface>(wrappedTaskOp);
}

void vpux::setDmaHwpIdAttribute(mlir::MLIRContext* ctx, VPUIP::DMATypeOpInterface& op, int32_t dmaHwpId) {
    auto dmaHwpIdAttrib = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed), dmaHwpId);
    op.setDmaHwpIdAttr(dmaHwpIdAttrib);
}

bool vpux::isDmaHwpUsedInVPURT(mlir::func::FuncOp& func) {
    bool dmaHwpEnabled = false;
    func->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::interrupt();
        }

        auto op = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());
        if (op && op.getDmaHwpIdAttr() != nullptr) {
            dmaHwpEnabled = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return dmaHwpEnabled;
}

bool vpux::isDmaHwpUsedInVPURT(mlir::ModuleOp& module) {
    if (vpux::VPU::getArch(module) <= vpux::VPU::ArchKind::VPUX37XX) {
        return false;
    }
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netOp, func);
    return vpux::isDmaHwpUsedInVPURT(func);
}
