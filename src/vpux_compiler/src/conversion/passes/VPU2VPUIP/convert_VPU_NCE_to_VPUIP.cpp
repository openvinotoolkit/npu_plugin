//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/IR/BlockAndValueMapping.h>
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
                     ppeAttr.quant_scale(), input1MultList, input2MultList, ppeAttr.fp_prelu_alpha());
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
        const auto outDpuStart = {offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()],
                                  offsets[Dims4D::Act::C.ind()]};
        const auto outDpuEnds = {ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()]};

        nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                         dpuTaskOp.pad(), dpuTaskOp.mpe_mode(), dpuTaskOp.cluster_idAttr());
    }
}

//
// Create VPUIP.NCEClusterTask and ensure sparse types interact with the operation as individual buffers
//

mlir::Value createNCEClusterTask(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                 mlir::Value weights, mlir::Value weightsTable, mlir::Value instructionListTable,
                                 mlir::Value activationWindow, ArrayRef<mlir::Value> outputBuffs,
                                 vpux::VPUIP::NCETaskType taskType, mlir::ArrayAttr kernelSizeAttr,
                                 mlir::ArrayAttr kernelStridesAttr, vpux::VPU::PaddingAttr kernelPaddingAttr,
                                 mlir::IntegerAttr activationWindowChannelLengthAttr, mlir::Region& workloads,
                                 mlir::UnitAttr isSuperdenseAttr = nullptr, VPU::PPETaskAttr ppeAttr = nullptr,
                                 mlir::Attribute dpuCostAttr = nullptr, mlir::BoolAttr isInplace = nullptr,
                                 Logger log = Logger::global()) {
    const auto getIndividualBuffers = [&](mlir::Value value) {
        mlir::Value data = value;
        mlir::Value sparsityMap = nullptr;
        mlir::Value seTable = nullptr;
        if (value != nullptr && value.getType().isa<VPUIP::SparseBufferType>()) {
            auto ungroupedOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(loc, value);
            data = ungroupedOp.data();
            sparsityMap = ungroupedOp.sparsityMap();
            seTable = ungroupedOp.storageElementTable();
        }
        return std::make_tuple(data, sparsityMap, seTable);
    };

    mlir::Value inputData, inputSparsityMap, inputSETable;
    std::tie(inputData, inputSparsityMap, inputSETable) = getIndividualBuffers(input);

    mlir::Value weightsData, weightsSparsityMap;
    std::tie(weightsData, weightsSparsityMap, std::ignore) = getIndividualBuffers(weights);

    mlir::Value outputBuffData = outputBuffs[0];
    mlir::Value outputBuffSparsityMap = (outputBuffs.size() > 1) ? outputBuffs[1] : nullptr;

    auto nceClusterTask = rewriter.create<VPUIP::NCEClusterTaskOp>(
            loc, inputData, inputSparsityMap, inputSETable, weightsData, weightsSparsityMap, weightsTable,
            instructionListTable, activationWindow, inputData, inputSparsityMap, inputSETable, outputBuffData,
            outputBuffSparsityMap, outputBuffData, outputBuffSparsityMap, nullptr, taskType, kernelSizeAttr,
            kernelStridesAttr, kernelPaddingAttr, activationWindowChannelLengthAttr,
            /*is_continued=*/nullptr,
            /*cm_sp_pattern=*/nullptr,
            /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr,
            /*input_channels_compression=*/nullptr, isSuperdenseAttr, isInplace,
            /*input_se_size=*/nullptr,
            /*output_se_size=*/nullptr);

    addDPUTasks(log, nceClusterTask, rewriter, workloads);

    if (ppeAttr != nullptr) {
        addPPETask(log, rewriter, nceClusterTask, ppeAttr);
    }

    if (dpuCostAttr != nullptr) {
        nceClusterTask->setAttr(DPUCost, dpuCostAttr);
    }

    if (nceClusterTask.output_sparsity_map() != nullptr) {
        auto groupedOp = rewriter.create<VPUIP::GroupSparseBufferOp>(loc, nceClusterTask.output(),
                                                                     nceClusterTask.output_sparsity_map());
        return groupedOp.output();
    }

    return nceClusterTask.output();
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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    const auto taskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.ppe().hasValue() ? origOp.ppeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), newArgs.input(), newArgs.filter(),
                                      newArgs.weightsTable(), newArgs.instructionListTable(),
                                      newArgs.activationWindow(), outputBuffers, taskType, kernelSizeAttr,
                                      origOp.strides(), origOp.padAttr(), origOp.activation_window_channel_lengthAttr(),
                                      origOp.workloads(), /*isSuperdenseAttr=*/nullptr, ppeAttr, dpuCostAttr,
                                      /*isInplace=*/nullptr, _log);

    rewriter.replaceOp(origOp, nceOp);

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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.ppe().hasValue() ? origOp.ppeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.input(), /*weights=*/nullptr, newArgs.weightsTable(),
            /*instruction_list_table=*/nullptr, newArgs.activationWindow(), outputBuffers, VPUIP::NCETaskType::MAXPOOL,
            origOp.kernel_size(), origOp.strides(), origOp.pad(), origOp.activation_window_channel_lengthAttr(),
            origOp.workloads(), /*isSuperdenseAttr=*/nullptr, ppeAttr, dpuCostAttr, /*isInplace=*/nullptr, _log);

    rewriter.replaceOp(origOp, nceOp);

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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.ppe().hasValue() ? origOp.ppeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), newArgs.input(), /*weights=*/nullptr,
                                      /*weights_table=*/nullptr,
                                      /*instruction_list_table=*/nullptr, /*activation_window=*/nullptr, outputBuffers,
                                      VPUIP::NCETaskType::AVEPOOL, origOp.kernel_size(), origOp.strides(), origOp.pad(),
                                      /*activation_window_channel_length=*/nullptr, origOp.workloads(),
                                      /*isSuperdenseAttr=*/nullptr, ppeAttr, dpuCostAttr, /*isInplace=*/nullptr, _log);

    rewriter.replaceOp(origOp, nceOp);

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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    auto ppeAttr = origOp.ppe().hasValue() ? origOp.ppeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.input(), newArgs.filter(), newArgs.weightsTable(),
            newArgs.instructionListTable(), newArgs.activationWindow(), outputBuffers, VPUIP::NCETaskType::DWCONV,
            kernelSizeAttr, origOp.strides(), origOp.pad(), origOp.activation_window_channel_lengthAttr(),
            origOp.workloads(), /*isSuperdenseAttr=*/nullptr, ppeAttr, dpuCostAttr, /*isInplace=*/nullptr, _log);

    rewriter.replaceOp(origOp, nceOp);

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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(this->getContext(), static_cast<int32_t>(0));
    auto ppeAttr = origOp.ppe().hasValue() ? origOp.ppeAttr() : nullptr;
    VPUX_THROW_UNLESS(ppeAttr != nullptr, "Eltwise operation should always have PPE info");
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp =
            createNCEClusterTask(rewriter, origOp->getLoc(), newArgs.input1(), newArgs.input2(),
                                 /*weightsTable=*/nullptr,
                                 /*instruction_table_list=*/nullptr,
                                 /*activation_window=*/nullptr, outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                 /*kernel_size=*/nullptr,
                                 /*kernel_strides=*/nullptr,
                                 /*kernel_padding=*/nullptr, activation_window_channel_length, origOp.workloads(),
                                 /*isSuperdenseAttr=*/nullptr, ppeAttr, dpuCostAttr, origOp.is_inplaceAttr(), _log);

    rewriter.replaceOp(origOp, nceOp);

    return mlir::success();
}

//
// PermuteQuantizeRewriter
//

class PermuteQuantizeRewriter final : public mlir::OpConversionPattern<VPU::NCEPermuteQuantizeOp> {
public:
    PermuteQuantizeRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEPermuteQuantizeOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEPermuteQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    VPU::PPETaskAttr composePPETask(const vpux::NDTypeInterface outType) const;
    Logger _log;
};

VPU::PPETaskAttr PermuteQuantizeRewriter::composePPETask(const vpux::NDTypeInterface outType) const {
    auto outElemType = outType.getElementType();
    auto uniformQElemType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const int64_t lReluMult = 1;
    const int64_t lReluShift = 0;
    // int64_t data types with int32_t numeric limits are used intentionally.
    // Schema requires i32 numeric limits to by-pass clamping, while PPETaskOp stores these values as i64.
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    if (uniformQElemType == nullptr) {
        return getPPETaskAttr(outType.getContext(), vpux::VPU::PPEMode::ADD, clampLow, clampHigh, lReluMult, lReluShift,
                              ArrayRef<double>{0.5});
    }
    const auto scale = uniformQElemType.getScale();
    // Since element-wise add sums the input with itself, output scale must be doubled.
    const auto newScale = static_cast<double>(scale * 2.0);
    clampLow = uniformQElemType.getStorageTypeMin();
    clampHigh = uniformQElemType.getStorageTypeMax();
    return getPPETaskAttr(outType.getContext(), vpux::VPU::PPEMode::ADD, clampLow, clampHigh, lReluMult, lReluShift,
                          ArrayRef<double>{1.0 / newScale});
}

mlir::LogicalResult PermuteQuantizeRewriter::matchAndRewrite(VPU::NCEPermuteQuantizeOp origOp, OpAdaptor newArgs,
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

    const auto outputBuffers = allocateBuffers(_log, origOp.getLoc(), rewriter, *typeConverter, {origOp.output()},
                                               /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(this->getContext(), static_cast<int32_t>(0));

    // Add PPE task to rescale output.
    auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto ppeTaskAttr = composePPETask(outType);

    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (origOp.isSuperdense()) {
        isSuperdenseAttr = mlir::UnitAttr::get(this->getContext());
    }

    _log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), newArgs.input(), newArgs.input(),
                                      /*weightsTable=*/nullptr,
                                      /*instruction_table_list=*/nullptr,
                                      /*activation_window=*/nullptr, outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                      /*kernel_size=*/nullptr,
                                      /*kernel_strides=*/nullptr,
                                      /*kernel_padding=*/nullptr, activation_window_channel_length, origOp.workloads(),
                                      isSuperdenseAttr, ppeTaskAttr, dpuCostAttr,
                                      /*isInplace=*/nullptr, _log);

    rewriter.replaceOp(origOp, nceOp);

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
    patterns.add<PermuteQuantizeRewriter>(typeConverter, &ctx, _log);

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
