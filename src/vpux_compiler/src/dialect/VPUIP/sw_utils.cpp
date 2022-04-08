//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

namespace vpux {
namespace VPUIP {
namespace {

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }
    return innerModule;
}

}  // namespace

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, StringRef kernelEntryName,
                                          StringRef kernelSourceFileName, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    auto vpuswModule = getVPUSWModule(module, log);

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().getValue(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName)) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto builtInOp = innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    builtInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    builtInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    builtInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName = static_cast<const SmallString>("nnActEntry");

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::FuncOp>(runtimeKernelName);
    if (runtimeKernelFunction) {
        log.trace("Found builtin function: {0}", runtimeKernelName);
        return;
    }

    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    // creating runtime kernel function
    const auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto runtimeFunctionOp =
            innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    runtimeFunctionOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, runtimeKernelEntryName));

    log.trace("Added runtime kernel function: {0}", runtimeKernelEntryName);

    // creating name symbol
    auto runtimeFlatSym = mlir::SymbolRefAttr::get(ctx, runtimeKernelName);
    auto runtimeSym = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().getValue(), {runtimeFlatSym});

    static constexpr int64_t defaultStackSize = 4096;

    // TODO: always extract num shaves info from VPURT::SW.Runtime, which can be extracted from module
    const auto maxShaves = 4;
    SmallVector<int64_t> stacksArray(maxShaves, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
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
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);

    auto argsAttr = args.empty() ? nullptr : mlir::ArrayAttr::get(ctx, args);
    swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands), argsAttr);
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log) {
    auto& bodyRegion = swKernelOp.body();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    OpBuilderLogger builderLog(log);
    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            auto input = std::get<0>(arg);
            auto outputBuf = std::get<1>(arg);
            swKernelBlock.addArgument(input.getType(), input.getLoc());
            swKernelBlock.addArgument(outputBuf.getType(), outputBuf.getLoc());
        }
    };
    addBlockArgs(zip(swKernelOp.inputs(), swKernelOp.output_buffs()));

    auto numBlockArgs = swKernelBlock.getNumArguments();
    auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
    VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto tileNum = numBlockArgs / numSwKernelRunArgs;

    // pack input/outputs and constants into several sw_kernel_run calls
    for (auto tileIdx : irange(tileNum)) {
        auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
        for (auto argIdx : irange(numSwKernelRunArgs)) {
            newRunOp->setOperand(argIdx, swKernelBlock.getArgument(tileIdx + argIdx * tileNum));
        }
        log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
    }
}

// Check whether SwKernelOp supports tiling.
bool isSwOpTilingSupported(VPU::SWOpInterface swOp) {
    if (mlir::isa<VPU::MVNOp>(swOp)) {
        return true;
    }
    return false;
}

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelOp.kernelFunctionAttr());
    VPUX_THROW_WHEN(kernelFunc == nullptr, "Cannot find kernel function symbol at '{0}'", swKernelOp->getLoc());
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    VPUX_THROW_WHEN(kernelEntryPoint == nullptr, "Cannot find kernel entry point at '{0}'", swKernelOp->getLoc());
    return kernelEntryPoint.getValue();
}

// Check whether SwKernelOp supports tiling.
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_SUPPORTING_TILING, kernelEntryName) != SW_KERNELS_SUPPORTING_TILING.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp support discontinuous input/output.
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_SUPPORTING_STRIDE, kernelEntryName) != SW_KERNELS_SUPPORTING_STRIDE.end()) {
        return true;
    }
    return false;
}
}  // namespace VPUIP
}  // namespace vpux
