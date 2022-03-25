//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/profiling.hpp"

using namespace vpux;

mlir::BlockArgument vpux::addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::FuncOp& netFunc, IE::CNNNetworkOp& netOp,
                                                mlir::MemRefType outputType, StringRef name) {
    //
    // Declare and create additional output from network
    //
    auto funcType = netFunc.getType();
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
    userInfoBuilder.create<IE::DataInfoOp>(mlir::NameLoc::get(mlir::Identifier::get("profilingDataOutputInfo", ctx)),
                                           mlir::StringAttr::get(ctx, name), mlir::TypeAttr::get(outputUserResult));

    return netFunc.getBody().front().addArgument(outputType);
}
