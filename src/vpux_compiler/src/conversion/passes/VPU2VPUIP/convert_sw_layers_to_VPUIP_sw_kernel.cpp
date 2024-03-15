//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/conversion/rewriters/VPU2VPUIP/sw_rewriter.hpp>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace {

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

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");
    if (mlir::failed(vpux::bufferizeSoftwareLayer(rewriter, _module, origOp, newOperands, std::nullopt, *typeConverter,
                                                  _log))) {
        return mlir::failure();
    }
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

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    if (mlir::failed(vpux::bufferizeNceClusterTilingSoftwareLayer(rewriter, _module, origOp, newOperands, std::nullopt,
                                                                  *typeConverter, _log))) {
        return mlir::failure();
    }
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

            auto inputElemType = convertOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
            auto outputElemType = convertOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();

            return !checkElementType(inputElemType) || !checkElementType(outputElemType);
        }
        if (auto stridedSliceOp = mlir::dyn_cast<VPU::StridedSliceOp>(op)) {
            auto attrToVector = [&](mlir::ArrayAttr attr) {
                return to_small_vector(parseIntArrayAttr<uint32_t>(attr));
            };
            const auto greaterThanOne = [](auto dim) {
                return dim > 1;
            };
            const auto stridesVec = attrToVector(stridedSliceOp.getStridesAttrAttr());
            const auto beginsVec = attrToVector(stridedSliceOp.getBeginsAttrAttr());

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
}

}  // namespace

//
// createConvertSWLayers2VPUIPSWKernelPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPSWKernelPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPSWKernelPass>(log);
}
