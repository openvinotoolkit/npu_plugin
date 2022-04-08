//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace {

mlir::Value allocateResult(mlir::Location loc, mlir::PatternRewriter& rewriter, mlir::TypeConverter& typeConverter,
                           mlir::Type type, vpux::IndexedSymbolAttr memSpace) {
    auto bufferType = typeConverter.convertType(type).cast<vpux::NDTypeInterface>();
    auto resultBufferType = bufferType.changeMemSpace(memSpace);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(loc, resultBufferType.cast<mlir::MemRefType>());
    return allocOp.memref();
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log) {
    OpBuilderLogger builderLog(log);

    SmallString builtInFunctionName{"builtin_"};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    auto& args = kernelInfo.args;

    SmallVector<mlir::Type> inputTypes;
    std::transform(operands.begin(), operands.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(results.begin(), results.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [](mlir::Attribute arg) {
        return arg.getType();
    });

    return VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelInfo.entryName,
                                        kernelInfo.sourceFileName, log);
}

//
// SoftwareLayerRewriter
//

class SoftwareLayerRewriter final : public mlir::OpInterfaceConversionPattern<VPUIP::SoftwareLayerOpInterface> {
public:
    SoftwareLayerRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpInterfaceConversionPattern<VPUIP::SoftwareLayerOpInterface>(typeConverter, ctx),
              _module(module),
              _log(log) {
        setDebugName("SoftwareLayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SoftwareLayerOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    mlir::ModuleOp _module;
    Logger _log;

private:
};

mlir::LogicalResult SoftwareLayerRewriter::matchAndRewrite(VPUIP::SoftwareLayerOpInterface origOp,
                                                           ArrayRef<mlir::Value> newOperands,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got operation {0} at {1}", origOp->getName(), origOp->getLoc());

    auto layerOp = mlir::dyn_cast<VPU::LayerOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} is not a layer.", origOp);

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto* ctx = getContext();

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    SmallVector<mlir::Value> cmxOperands;
    for (auto&& operand : newOperands) {
        _log.trace("Create CMX buffer and copy operation for input: {0}", operand.getLoc());
        const auto type = operand.getType().cast<vpux::NDTypeInterface>().changeMemSpace(memSpaceCMX);
        auto cmxAllocOp = rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), type.cast<mlir::MemRefType>());
        auto copyOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), operand, cmxAllocOp.memref());
        cmxOperands.push_back(copyOp.output());
    }

    SmallVector<mlir::Value> cmxResults;
    for (auto&& result : origOp->getResults()) {
        _log.trace("Create CMX buffer for output: {0}", result.getLoc());
        const auto outputBuffer =
                allocateResult(origOp.getLoc(), rewriter, *typeConverter, result.getType(), memSpaceCMX);
        cmxResults.push_back(outputBuffer);
    }

    VPUIP::createRuntimeKernelDefinition(_module, _log.nest(), VPU::getArch(origOp));

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto builtInFunction =
            createBuiltInFunction(_module, layerOp, cmxOperands, cmxResults, origOp.getKernelInfo(), _log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), cmxOperands, cmxResults, builtInFunction,
                                                         getIntAttr(ctx, tileIndex));

    initSwKernel(swKernelOp, cmxOperands, cmxResults, origOp.getKernelInfo().args, _log.nest());

    _log.trace("Added kernel operation: {0}", swKernelOp);

    SmallVector<mlir::Value> outputDmaResults;
    SmallVector<mlir::Value> ddrResults;
    for (auto&& result : swKernelOp.results()) {
        _log.trace("Create DDR buffer for output: {0}", result.getLoc());
        const auto outputBuffer = allocateResult(origOp.getLoc(), rewriter, *typeConverter, result.getType(), nullptr);
        ddrResults.push_back(outputBuffer);
    }
    std::transform(ddrResults.begin(), ddrResults.end(), swKernelOp.results().begin(),
                   std::back_inserter(outputDmaResults), [&](const auto& outputBuff, const auto& swKernelResult) {
                       auto copyOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), swKernelResult, outputBuff);
                       return copyOp.output();
                   });

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
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("ConvertSWLayers2VPUIPPass enabled only for VPUX37XX device. Got: {0}", arch);
        return;
    }

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto convertOp = mlir::dyn_cast<VPU::ConvertOp>(op)) {
            const auto checkElementType = [](mlir::Type elemType) {
                return elemType.isF16() || elemType.isF32() || elemType.isSignedInteger(8) ||
                       elemType.isSignedInteger(32) || elemType.isSignedInteger(64) || elemType.isUnsignedInteger(8);
            };
            auto inputElemType = convertOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
            auto outputElemType = convertOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
            return !checkElementType(inputElemType) || !checkElementType(outputElemType);
        }
        return !mlir::isa<VPUIP::SoftwareLayerOpInterface>(op);
    });
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SoftwareLayerRewriter>(typeConverter, &ctx, module, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
    // Find illegal software ops which are not supported by VPUX37XX
    auto isSoftwareVPUOp = [](mlir::Operation* op) {
        return !(mlir::isa<VPU::NCEOpInterface>(op) || mlir::isa<VPU::ViewLikeOpInterface>(op) ||
                 mlir::isa<VPU::GroupedViewLikeOpInterface>(op) ||
                 mlir::isa<VPU::NCEClusterTilingOp, VPU::DPUWorkloadOp, VPU::YieldOp, VPU::CopyOp, VPU::TileOp,
                           VPU::ExpandOp, VPU::ExpandDilatedOp>(op) ||
                 mlir::isa<VPU::StubOp>(op));
    };
    auto foundUnsupportedKernel = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect()->getNamespace() == VPU::VPUDialect::getDialectNamespace() && isSoftwareVPUOp(op)) {
            _log.trace("Missing SW.Kernel operator for '{0}' VPU operation at '{1}'", op->getName(), op->getLoc());
            foundUnsupportedKernel = true;
        }
    });
    VPUX_THROW_WHEN(foundUnsupportedKernel, "Unsupported software kernel found");
}

}  // namespace

//
// createConvertSWLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPPass>(log);
}
