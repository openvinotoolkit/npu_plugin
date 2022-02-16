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
    VPUX_THROW_WHEN(netOp.profilingOutputsInfo().empty(), "Cound not add profiling output: no region added");

    const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();

    // Adding output to the user info
    auto outputUserResult =
            getTensorType(ndOutputType.getShape(), ndOutputType.getElementType(), ndOutputType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.profilingOutputsInfo().front().front());
    userInfoBuilder.create<IE::DataInfoOp>(mlir::NameLoc::get(mlir::Identifier::get("profilingDataOutputInfo", ctx)),
                                           mlir::StringAttr::get(ctx, name), mlir::TypeAttr::get(outputUserResult));

    return netFunc.getBody().front().addArgument(outputType);
}
