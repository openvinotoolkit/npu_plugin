//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionName)) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto builtInOp =
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    builtInOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    builtInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    builtInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::VPU::ArchKind /*arch*/) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName = static_cast<const SmallString>("nnActEntry");

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(runtimeKernelName);
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
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

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
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(swKernelOp.inputs());
    addBlockArgs(swKernelOp.output_buffs());

    auto numBlockArgs = swKernelBlock.getNumArguments();
    auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
    VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
    VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto tileNum = numBlockArgs / numSwKernelRunArgs;

    VPUX_THROW_UNLESS(swKernelOp.inputs().size() % tileNum == 0 && swKernelOp.results().size() % tileNum == 0,
                      "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto numSwKernelRunInputs = swKernelOp.inputs().size() / tileNum;
    auto numSwKernelRunOutputs = swKernelOp.results().size() / tileNum;

    // pack input/outputs and constants into several sw_kernel_run calls
    // For example: For Operation that has 2 inputs, 1 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First intput with 1th tile
    //         [INPUT1_TILE0] as %arg1: Second intput with 1th tile
    //         [INPUT0_TILE1] as %arg2: First intput with 2th tile
    //         [INPUT1_TILE1] as %arg3: Second intput with 2th tile
    // outputs:[OUTPUT_TILE0] as %arg4: Output of 1th tile
    //         [OUTPUT_TILE1] as %arg5: Output of 2th tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg1, %arg4)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg2, %arg3, %arg5)
    for (auto tileIdx : irange(tileNum)) {
        auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
        for (auto argInputIdx : irange(numSwKernelRunInputs)) {
            newRunOp->setOperand(argInputIdx, swKernelBlock.getArgument(tileIdx * numSwKernelRunInputs + argInputIdx));
        }

        for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
            newRunOp->setOperand(numSwKernelRunInputs + argOutputIdx,
                                 swKernelBlock.getArgument(tileNum * numSwKernelRunInputs +
                                                           tileIdx * numSwKernelRunOutputs + argOutputIdx));
        }

        log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
    }
}

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.kernelFunctionAttr());
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
    // SubView can be used for Softmax because it is always tilied on the highest dimension.
    if (kernelEntryName == "singleShaveSoftmax" ||
        llvm::find(SW_KERNELS_SUPPORTING_STRIDE, kernelEntryName) != SW_KERNELS_SUPPORTING_STRIDE.end()) {
        return true;
    }
    return false;
}

namespace {
// reverse int array attribute from the physical order
SmallVector<int64_t> reverseIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revDstInd] = origArray[revSrcInd];
    }
    return permArray;
}

// permute int array attribute in the physical order
static SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, SmallVector<int64_t> origArray) {
    const auto origPerm = inOrder.toPermutation();
    SmallVector<int64_t> permArray(origArray.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

InputTiling backInferInterpolateSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                  Logger log) {
    auto swKernelRuns = swKernelOp.body().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.attrs().hasValue(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.attrs().getValue();
    auto inOrder = swKernelOp.inputs()[0].getType().dyn_cast<vpux::NDTypeInterface>().getDimsOrder();

    const auto coordMode = static_cast<IE::InterpolateCoordMode>(attrs[1].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto initialInputDims = reverseIntArrayAttr(inOrder, attrs[5].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(inOrder, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(inOrder, attrs[9].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(inOrder, attrs[10].dyn_cast<mlir::ArrayAttr>());
    return vpux::backInferInterpolateTile(outputTile, initialInputDims, initialOutputDims, initialInputOffset,
                                          initialOutputOffset, coordMode, log);
}

SmallVector<mlir::Attribute> getInterpolateSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                       ArrayRef<mlir::Attribute> origAttr,
                                                                       const TilingInfo& inputTiling,
                                                                       const TileInfo& outTile, Logger log) {
    log.trace("update attrs for SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    // Get output tile against the original output
    auto kernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.attrs().getValue();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    auto dim = swKernelOp.inputs()[0].getType().dyn_cast<vpux::NDTypeInterface>().getDimsOrder();
    TileInfo inputTile = inputTiling.tiles[0];
    const auto initialInputDims = reverseIntArrayAttr(dim, attrs[5].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(dim, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(dim, attrs[9].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(dim, attrs[10].dyn_cast<mlir::ArrayAttr>());
    const auto localInputOffset = to_small_vector(inputTile.offsets);
    const auto localOutputOffset = to_small_vector(outTile.offsets);
    SmallVector<int64_t> inputTileOffset;
    SmallVector<int64_t> outputTileOffset;
    std::transform(localInputOffset.begin(), localInputOffset.end(), initialInputOffset.begin(),
                   std::back_inserter(inputTileOffset), std::plus<int64_t>());
    std::transform(localOutputOffset.begin(), localOutputOffset.end(), initialOutputOffset.begin(),
                   std::back_inserter(outputTileOffset), std::plus<int64_t>());
    auto newInputTiling = inputTiling;
    newInputTiling.tiles[0].offsets = Shape(inputTileOffset);
    auto newOutputTile = outTile;
    newOutputTile.offsets = Shape(outputTileOffset);
    newAttrs[9] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, inputTileOffset));
    newAttrs[10] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, outputTileOffset));
    return newAttrs;
}
}  // namespace

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveInterpolate") {
        return backInferInterpolateSwKernelInputTile(swKernelOp, outputTile, log);
    }
    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.inputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log) {
    log.trace("Update SwKernel attrs after tiling at '{0}'", swKernelOp->getLoc());
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveInterpolate") {
        return getInterpolateSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, inputTiling, outTile, log);
    } else {
        return SmallVector<mlir::Attribute>(origAttr.begin(), origAttr.end());
    }
}
}  // namespace VPUIP
}  // namespace vpux
