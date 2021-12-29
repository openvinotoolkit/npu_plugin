//
// Copyright 2021 Intel Corporation.
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

// #include "vpux/compiler/dialect/VPUIP/utils.hpp"

// #include "vpux/compiler/core/attributes/shape.hpp"
// #include "vpux/compiler/core/layers.hpp"
// #include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"

namespace vpux {
namespace VPUIP {

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module,
                                          StringRef builtInFunctionName, ArrayRef<mlir::Type> inputTypes,
                                          StringRef kernelEntryName, StringRef  kernelSourceFileName,
                                          const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, innerModule.getName().getValue(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    auto prebuiltFunction = innerModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName);
    if (prebuiltFunction) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);
    auto buildInOp = innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    buildInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log) {
    // Function name
    SmallString builtInFunctionName{"builtin_"};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    // Original Operation in/out
    auto opInputs = origOp.getInputs();
    auto opResults = origOp->getResults();

    // Kernel Info attributes
    auto& args = kernelInfo.args;
    auto kernelEntryName = kernelInfo.entryName;
    auto kernelSourceFileName = kernelInfo.sourceFileName;

    // Input Types
    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    SmallVector<mlir::Type> inputTypes;
    std::transform(opInputs.begin(), opInputs.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(opResults.begin(), opResults.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [](mlir::Attribute arg) {
        return arg.getType();
    });

    return createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelEntryName, kernelSourceFileName, log);
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.body();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding args of IERT operation as constants
    SmallVector<mlir::arith::ConstantOp> constantArgs;
    for (auto&& arg : args) {
        constantArgs.push_back(swKernelBlockBuilder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
    }

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);
    fetchOperands(constantArgs);

    swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands));
}

}  // namespace VPUIP
}  // namespace vpux
