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
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
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

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    return innerModule;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    auto vpuswModule = getVPUSWModule(module, log);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, innerModule.getName().getValue(), {builtInFlatFunction});
    auto prebuiltFunction = innerModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName);
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(innerModule.getBody(), &builderLog);
void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName{"nnActEntry"};

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

    SmallVector<int64_t> stacksArray(4, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
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

    // first creating management kernel definition
    createRuntimeKernelDefinition(_module, _log.nest());

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto builtInFunction = VPUIP::createBuiltInFunction(_module, layerOp, origOp.getKernelInfo(), _log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), inputCMXTensors, outputCMXTensors,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    _log.trace("Added kernel operation: {0}", swKernelOp);
    VPUIP::initSwKernel(swKernelOp, inputCMXTensors, outputCMXTensors, origOp.getKernelInfo().args, _log.nest());

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
