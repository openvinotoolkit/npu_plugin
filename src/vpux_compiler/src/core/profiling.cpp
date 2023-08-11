//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Visitors.h>
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

using namespace vpux;

mlir::BlockArgument vpux::addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc,
                                                IE::CNNNetworkOp& netOp, mlir::MemRefType outputType, StringRef name) {
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
    VPUX_THROW_WHEN(netOp.profilingOutputsInfo().empty(), "Could not add profiling output: no region added");

    const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();

    // Adding output to the user info
    auto outputUserResult =
            getTensorType(ndOutputType.getShape(), ndOutputType.getElementType(), ndOutputType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.profilingOutputsInfo().front().front());
    userInfoBuilder.create<IE::DataInfoOp>(mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profilingDataOutputInfo")),
                                           mlir::StringAttr::get(ctx, name), mlir::TypeAttr::get(outputUserResult));

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
    op.setDmaHwpId(dmaHwpIdAttrib);
}

bool vpux::isDmaHwpUsedInVPURT(mlir::func::FuncOp& func) {
    bool dmaHwpEnabled = false;
    func->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::interrupt();
        }

        auto op = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());
        if (op && op.getDmaHwpId() != nullptr) {
            dmaHwpEnabled = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return dmaHwpEnabled;
}
