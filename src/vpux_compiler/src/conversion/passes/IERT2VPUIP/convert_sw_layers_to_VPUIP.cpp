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

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace {

mlir::memref::AllocOp createCMXTensor(mlir::Value source, mlir::PatternRewriter& rewriter) {
    auto type = source.getType().cast<mlir::MemRefType>();
    const auto dataTypeCMX = changeMemSpace(eraseTiledInfo(type), VPU::MemoryKind::CMX_NN);

    // TODO : how tile index should be used?
    return rewriter.create<mlir::memref::AllocOp>(source.getLoc(), dataTypeCMX);
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }

    SmallString builtInFunctionName{"builtin_"};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, innerModule.getName().getValue(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    auto prebuiltFunction = innerModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName);
    if (prebuiltFunction) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    auto& args = kernelInfo.args;
    auto opInputs = origOp.getInputs();
    auto opResults = origOp->getResults();

    SmallVector<mlir::Type> inputTypes;
    std::transform(opInputs.begin(), opInputs.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(opResults.begin(), opResults.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [](mlir::Attribute arg) {
        return arg.getType();
    });

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);
    auto buildInOp = innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    buildInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelInfo.entryName));
    buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelInfo.sourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

void convertTopKAttr(mlir::OpBuilder builder, mlir::MLIRContext* ctx, ArrayRef<mlir::Attribute> args,
                     SmallVector<mlir::arith::ConstantOp>& constantArgs) {
    for (auto&& arg : args) {
        if (arg.isa<mlir::IntegerAttr, mlir::FloatAttr>()) {
            constantArgs.push_back(builder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
        } else if (auto mode = arg.dyn_cast_or_null<IE::TopKModeAttr>()) {
            mlir::IntegerAttr intAttr;
            switch (mode.getValue()) {
            case IE::TopKMode::MAX:
                intAttr = getIntAttr(ctx, 0);
                break;
            case IE::TopKMode::MIN:
                intAttr = getIntAttr(ctx, 1);
                break;
            default:
                VPUX_THROW("Unknown TopKMode");
            }
            constantArgs.push_back(builder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), intAttr));
        } else if (auto sort = arg.dyn_cast_or_null<IE::TopKSortTypeAttr>()) {
            mlir::IntegerAttr intAttr;
            switch (sort.getValue()) {
            case IE::TopKSortType::NONE:
                intAttr = getIntAttr(ctx, 0);
                break;
            case IE::TopKSortType::SORT_INDICES:
                intAttr = getIntAttr(ctx, 1);
                break;
            case IE::TopKSortType::SORT_VALUES:
                intAttr = getIntAttr(ctx, 2);
                break;
            default:
                VPUX_THROW("Unknown TopKSortType");
            }
            constantArgs.push_back(builder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), intAttr));
        } else if (auto type = arg.dyn_cast_or_null<mlir::TypeAttr>()) {
            // TODO: Check actual type i32/i64.
            mlir::IntegerAttr intAttr;
            intAttr = getIntAttr(ctx, 0);
            constantArgs.push_back(builder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), intAttr));
        } else {
            VPUX_THROW("Unknown TopK Attribute");
        }
    }
}

void convertAttr(mlir::OpBuilder builder, mlir::MLIRContext* ctx, IERT::LayerOpInterface origOp,
                 ArrayRef<mlir::Attribute> args, SmallVector<mlir::arith::ConstantOp>& constantArgs) {
    if (mlir::isa<IERT::TopKOp>(origOp)) {
        convertTopKAttr(builder, ctx, args, constantArgs);
    } else {
        for (auto&& arg : args) {
            constantArgs.push_back(builder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
        }
    }
}

void initSwKernel(IERT::LayerOpInterface origOp, VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs,
                  mlir::ValueRange outputBuffs, ArrayRef<mlir::Attribute> args, const Logger& log) {
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
    convertAttr(swKernelBlockBuilder, ctx, origOp, args, constantArgs);

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

//
// SoftwareLayerRewriter
//

class SoftwareLayerRewriter final : public mlir::OpInterfaceRewritePattern<IERT::SoftwareLayerOpInterface> {
public:
    SoftwareLayerRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpInterfaceRewritePattern<IERT::SoftwareLayerOpInterface>(ctx), _module(module), _log(log) {
        setDebugName("SoftwareLayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::SoftwareLayerOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::ModuleOp _module;
    Logger _log;

private:
};

mlir::LogicalResult SoftwareLayerRewriter::matchAndRewrite(IERT::SoftwareLayerOpInterface origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto layerOp = mlir::dyn_cast<IERT::LayerOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} is not a layer.", origOp);

    _log.trace("Got operation {0}", origOp->getName());
    auto* ctx = getContext();

    //  creating input dma
    SmallVector<mlir::Value> inputCMXTensors;
    for (auto&& source : layerOp.getInputs()) {
        _log.trace("Create CMX buffer for input: {0}", source.getLoc());
        auto cmxTensor = createCMXTensor(source, rewriter);
        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), source, cmxTensor.memref());
        inputCMXTensors.push_back(copyOp.output());
    }

    // allocating output tensors
    SmallVector<mlir::Value> outputCMXTensors;
    for (auto&& output : layerOp.getOutputs()) {
        _log.trace("Create CMX buffer for output: {0}", output.getLoc());
        auto allocOp = createCMXTensor(output, rewriter);
        outputCMXTensors.push_back(allocOp.memref());
    }

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto builtInFunction = createBuiltInFunction(_module, layerOp, origOp.getKernelInfo(), _log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), inputCMXTensors, outputCMXTensors,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    _log.trace("Added kernel operation: {0}", swKernelOp);
    initSwKernel(layerOp, swKernelOp, inputCMXTensors, outputCMXTensors, origOp.getKernelInfo().args, _log.nest());

    SmallVector<mlir::Value> outputDmaResults;
    auto opOutputs = layerOp.getOutputs();
    //  creating post-dma
    std::transform(opOutputs.begin(), opOutputs.end(), swKernelOp.results().begin(),
                   std::back_inserter(outputDmaResults), [&](const auto& outputBuff, const auto& swKernelResult) {
                       auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), swKernelResult, outputBuff);
                       return copyOp.output();
                   });

    // setting output to be from DMA
    _log.trace("Replace origin op {0} with new outputs from SW Kernel {1}", origOp.getLoc(), outputDmaResults);
    rewriter.replaceOp(origOp, outputDmaResults);

    return mlir::success();
}

//
// ConvertLayers2VPUIPPass
//

class ConvertSWLayers2VPUIPPass final : public ConvertSWLayers2VPUIPBase<ConvertSWLayers2VPUIPPass> {
public:
    explicit ConvertSWLayers2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertSWLayers2VPUIPPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::MTL) {
        _log.trace("ConvertSWLayers2VPUIPPass enabled only for MTL device. Got: {0}", arch);
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        return !mlir::isa<IERT::SoftwareLayerOpInterface>(op);
    });
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<IERT::CopyOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SoftwareLayerRewriter>(&ctx, module, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPPass>(log);
}
