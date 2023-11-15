//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/conversion/rewriters/VPU2VPUIP/sw_rewriter.hpp>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/dma.hpp"
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

    SmallString builtInFunctionName{VPUIP::SW_KERNEL_NAME_PREFIX};
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
    auto clusterTiling = origOp->getParentOfType<VPU::NCEClusterTilingOp>();
    if (clusterTiling) {
        return mlir::failure();
    }

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
// NCEClusterTilingSoftwareLayerRewriter
//

class NCEClusterTilingSoftwareLayerRewriter final :
        public mlir::OpInterfaceConversionPattern<VPUIP::SoftwareLayerOpInterface> {
public:
    NCEClusterTilingSoftwareLayerRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                                          mlir::ModuleOp module, Logger log)
            : mlir::OpInterfaceConversionPattern<VPUIP::SoftwareLayerOpInterface>(typeConverter, ctx),
              _module(module),
              _log(log) {
        setDebugName("NCEClusterTilingSoftwareLayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SoftwareLayerOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    mlir::ModuleOp _module;
    Logger _log;

private:
};

mlir::LogicalResult NCEClusterTilingSoftwareLayerRewriter::matchAndRewrite(
        VPUIP::SoftwareLayerOpInterface origOp, ArrayRef<mlir::Value> newOperands,
        mlir::ConversionPatternRewriter& rewriter) const {
    auto clusterTiling = origOp->getParentOfType<VPU::NCEClusterTilingOp>();
    if (!clusterTiling) {
        return mlir::failure();
    }

    _log.trace("Got operation {0} at {1}", origOp->getName(), origOp->getLoc());

    auto layerOp = mlir::dyn_cast<VPU::LayerOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} is not a layer.", origOp);

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto* ctx = getContext();
    VPUIP::createRuntimeKernelDefinition(_module, _log.nest(), VPU::getArch(origOp));

    auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, origOp->getResults(),
                                         /*individualBuffers=*/true);
    // actual tile index will be corrected as part of unroll NCEClusterTiling pass, this index will be dropped
    const int64_t tileIndex = 0;
    auto builtInFunction =
            createBuiltInFunction(_module, layerOp, newOperands, outputBuffers, origOp.getKernelInfo(), _log.nest());
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), newOperands, outputBuffers, builtInFunction,
                                                         getIntAttr(ctx, tileIndex));
    initSwKernel(swKernelOp, newOperands, outputBuffers, origOp.getKernelInfo().args, _log.nest());
    rewriter.replaceOp(origOp, swKernelOp.results());

    return mlir::success();
}

//
// ConvertLayers2VPUIPPass
//

class ConvertSWLayers2VPUIPSWKernelPass final :
        public ConvertSWLayers2VPUIPSWKernelBase<ConvertSWLayers2VPUIPSWKernelPass> {
public:
    explicit ConvertSWLayers2VPUIPSWKernelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertSWLayers2VPUIPSWKernelPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();
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
            // If conversion can be done on DMA, mark it legal so later pass can lower this to ConvertDMAOp
            if (isConvertSupportedOnDMA<VPU::ConvertOp>(convertOp)) {
                return true;
            }
            return !checkElementType(inputElemType) || !checkElementType(outputElemType);
        }
        if (auto stridedSliceOp = mlir::dyn_cast<VPU::StridedSliceOp>(op)) {
            auto attrToVector = [&](mlir::ArrayAttr attr) {
                return to_small_vector(parseIntArrayAttr<uint32_t>(attr));
            };
            const auto greaterThanOne = [](auto dim) {
                return dim > 1;
            };
            const auto stridesVec = attrToVector(stridedSliceOp.strides_attrAttr());
            const auto beginsVec = attrToVector(stridedSliceOp.begins_attrAttr());

            const auto strideDimCount = llvm::count_if(stridesVec, greaterThanOne);
            const auto beginsDimCount = llvm::count_if(beginsVec, greaterThanOne);
            // if the stride dim equals to 1, the layer could be converted to strided DMA.
            return strideDimCount == 1 && beginsDimCount <= 1;
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
    patterns.add<NCEClusterTilingSoftwareLayerRewriter>(typeConverter, &ctx, module, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
    // Find illegal software ops which are not supported by VPUX37XX
    auto isSoftwareVPUOp = [](mlir::Operation* op) {
        return !(mlir::isa<VPU::NCEOpInterface>(op) || mlir::isa<VPU::SWOpInterface>(op) ||
                 mlir::isa<VPU::ViewLikeOpInterface>(op) || mlir::isa<VPU::GroupedViewLikeOpInterface>(op) ||
                 mlir::isa<VPU::NCEClusterTilingOp, VPU::DPUWorkloadOp, VPU::YieldOp, VPU::CopyOp, VPU::TileOp,
                           VPU::ExpandOp, VPU::UpsamplingOp, VPU::ExpandDilatedOp, VPU::StorageElementTableOp>(op) ||
                 mlir::isa<VPU::StubOp>(op) || mlir::isa<VPU::ShapeCastOp>(op) || mlir::isa<VPU::StridedSliceOp>(op));
    };
    auto foundUnsupportedKernel = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getDialect()->getNamespace() == VPU::VPUDialect::getDialectNamespace() && isSoftwareVPUOp(op)) {
            _log.error("Missing SW.Kernel operator for '{0}' VPU operation at '{1}'", op->getName(), op->getLoc());
            foundUnsupportedKernel = true;
        }
    });
    VPUX_THROW_WHEN(foundUnsupportedKernel, "Unsupported software kernel found");
}  // namespace

}  // namespace

//
// createConvertSWLayers2VPUIPSWKernelPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPSWKernelPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPSWKernelPass>(log);
}
