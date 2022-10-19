//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

void addPPETask(const Logger& log, mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceOp, VPU::PPETaskAttr ppeAttr) {
    log.nest().trace("Adding PPE task '{0}'", ppeAttr);

    const auto input1MultList =
            ppeAttr.in1_quant_mult() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.in1_quant_mult())))
                    : nullptr;
    const auto input2MultList =
            ppeAttr.in2_quant_mult() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.in2_quant_mult())))
                    : nullptr;
    const auto outputMultList =
            ppeAttr.quant_mult() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.quant_mult())))
                    : nullptr;
    const auto shiftList =
            ppeAttr.quant_shift() != nullptr
                    ? builder.getI64ArrayAttr(makeArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.quant_shift())))
                    : nullptr;
    nceOp.addPPETask(builder, ppeAttr.mode(), ppeAttr.clamp_low(), ppeAttr.clamp_high(), ppeAttr.lrelu_mult(),
                     ppeAttr.lrelu_shift(), outputMultList, shiftList, ppeAttr.quant_post_shift(),
                     ppeAttr.quant_scale(), input1MultList, input2MultList);
}

void addDPUTasks(const Logger& log, VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter,
                 mlir::Region& workloads) {
    log.nest().trace("Adding DPU tasks");

    for (auto dpuTaskOp : workloads.getOps<VPU::DPUWorkloadOp>()) {
        SmallVector<int64_t> ends;
        const auto offsets = parseIntArrayAttr<int64_t>(dpuTaskOp.offsets());
        const auto sizes = parseIntArrayAttr<int64_t>(dpuTaskOp.sizes());
        ends.reserve(sizes.size());

        llvm::transform(llvm::seq<size_t>(0, sizes.size()), std::back_inserter(ends), [&](size_t index) {
            return offsets[index] + sizes[index] - 1;
        });

        // as soon as we need workload_x, workload_y, workload_z coords
        const auto dpuStart = {offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()],
                               offsets[Dims4D::Act::C.ind()]};
        const auto dpuEnds = {ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()]};

        nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, dpuStart), getIntArrayAttr(rewriter, dpuEnds),
                         dpuTaskOp.pad(), dpuTaskOp.mpe_mode(), dpuTaskOp.cluster_idAttr());
    }
}

//
// Buffer allocation
//

mlir::Value allocateResult(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                           mlir::TypeConverter& typeConverter, mlir::Value output) {
    auto origType = output.getType();
    auto memRefType = typeConverter.convertType(origType);
    log.nest().trace("Allocating result buffer of type '{0}'", memRefType);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
    return allocOp.memref();
}

//
// ConvRewriter
//

class ConvRewriter final : public mlir::OpConversionPattern<VPU::NCEConvolutionOp> {
public:
    ConvRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEConvolutionOp>(typeConverter, ctx), _log(log) {
        setDebugName("ConvRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvRewriter::matchAndRewrite(VPU::NCEConvolutionOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto inOrder = DimsOrder::fromValue(newArgs.input());
    const auto isCMajor = inOrder == DimsOrder::NCHW;

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(_log, origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    const auto taskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), newArgs.filter(), newArgs.weightsTable(), newArgs.instructionListTable(),
            newArgs.activationWindow(),
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, taskType, kernelSizeAttr, origOp.strides(), origOp.padAttr(),
            origOp.activation_window_channel_lengthAttr());

    addDPUTasks(_log, nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(_log, rewriter, nceOp, origOp.ppeAttr());
    }

    if (origOp->hasAttr(DPUCost)) {
        nceOp->setAttr(DPUCost, origOp->getAttr(DPUCost));
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpConversionPattern<VPU::NCEMaxPoolOp> {
public:
    MaxPoolRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEMaxPoolOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEMaxPoolOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(VPU::NCEMaxPoolOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(_log, origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), /*weights=*/nullptr, newArgs.weightsTable(),
            /*instruction_list_table=*/nullptr, newArgs.activationWindow(),
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(),
            origOp.pad(), origOp.activation_window_channel_lengthAttr());

    addDPUTasks(_log, nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(_log, rewriter, nceOp, origOp.ppeAttr());
    }

    if (origOp->hasAttr(DPUCost)) {
        nceOp->setAttr(DPUCost, origOp->getAttr(DPUCost));
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpConversionPattern<VPU::NCEAveragePoolOp> {
public:
    AveragePoolRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEAveragePoolOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEAveragePoolOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(VPU::NCEAveragePoolOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(_log, origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), /*weights=*/nullptr, /*weights_table=*/nullptr,
            /*instruction_list_table=*/nullptr, /*activation_window=*/nullptr,
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, VPUIP::NCETaskType::AVEPOOL, origOp.kernel_size(), origOp.strides(),
            origOp.pad(), /*activation_window_channel_length=*/nullptr);

    addDPUTasks(_log, nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(_log, rewriter, nceOp, origOp.ppeAttr());
    }

    if (origOp->hasAttr(DPUCost)) {
        nceOp->setAttr(DPUCost, origOp->getAttr(DPUCost));
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// DepthwiseConvRewriter
//

class DepthwiseConvRewriter final : public mlir::OpConversionPattern<VPU::NCEDepthConvolutionOp> {
public:
    DepthwiseConvRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEDepthConvolutionOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEDepthConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthwiseConvRewriter::matchAndRewrite(VPU::NCEDepthConvolutionOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(_log, origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), newArgs.filter(), newArgs.weightsTable(), newArgs.instructionListTable(),
            newArgs.activationWindow(),
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, VPUIP::NCETaskType::DWCONV, kernelSizeAttr, origOp.strides(), origOp.pad(),
            origOp.activation_window_channel_lengthAttr());

    addDPUTasks(_log, nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(_log, rewriter, nceOp, origOp.ppeAttr());
    }

    if (origOp->hasAttr(DPUCost)) {
        nceOp->setAttr(DPUCost, origOp->getAttr(DPUCost));
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// EltwiseRewriter
//

class EltwiseRewriter final : public mlir::OpConversionPattern<VPU::NCEEltwiseOp> {
public:
    EltwiseRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEEltwiseOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EltwiseRewriter::matchAndRewrite(VPU::NCEEltwiseOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(_log, origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(this->getContext(), static_cast<int32_t>(0));

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), newArgs.input1(), newArgs.input2(),
                                                          /*weightsTable=*/nullptr,
                                                          /*instruction_table_list=*/nullptr,
                                                          /*activation_window=*/nullptr,
                                                          /*parent_input=*/newArgs.input1(),
                                                          /*parent_output=*/outputBuffer,
                                                          /*output_buff=*/outputBuffer, VPUIP::NCETaskType::ELTWISE,
                                                          /*kernel_size=*/nullptr,
                                                          /*kernel_strides=*/nullptr,
                                                          /*kernel_padding=*/nullptr, activation_window_channel_length);

    //
    // Create DPU sub-task
    //

    addDPUTasks(_log, nceOp, rewriter, origOp.workloads());

    VPUX_THROW_UNLESS(origOp.ppe().hasValue(), "Eltwise operation should always have PPE info");

    addPPETask(_log, rewriter, nceOp, origOp.ppeAttr());

    if (origOp->hasAttr(DPUCost)) {
        nceOp->setAttr(DPUCost, origOp->getAttr(DPUCost));
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// ConvertVPUNCEToVPUIPPass
//

class ConvertVPUNCEToVPUIPPass final : public ConvertVPUNCEToVPUIPBase<ConvertVPUNCEToVPUIPPass> {
public:
    explicit ConvertVPUNCEToVPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertVPUNCEToVPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp,
                        VPU::NCEEltwiseOp>();
    // NCEClusterTiling will be handled in follow up pass (convertNCEClusterTilingToVPUIP pass)
    target.addLegalOp<VPU::NCEClusterTilingOp>();
    target.addLegalOp<VPU::YieldOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvRewriter>(typeConverter, &ctx, _log);
    patterns.add<DepthwiseConvRewriter>(typeConverter, &ctx, _log);
    patterns.add<MaxPoolRewriter>(typeConverter, &ctx, _log);
    patterns.add<AveragePoolRewriter>(typeConverter, &ctx, _log);
    patterns.add<EltwiseRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUNCEToVPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUNCEToVPUIPPass(Logger log) {
    return std::make_unique<ConvertVPUNCEToVPUIPPass>(log);
}
