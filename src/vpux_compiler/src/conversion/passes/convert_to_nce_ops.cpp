//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvertToNCEOpsPass
//

class ConvertToNCEOpsPass final : public ConvertToNCEOpsBase<ConvertToNCEOpsPass> {
public:
    ConvertToNCEOpsPass(Logger log);

public:
    class ConvRewrite;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

ConvertToNCEOpsPass::ConvertToNCEOpsPass(Logger log): _log(log) {
    _log.setName(Base::getArgumentName());
}

static std::tuple<mlir::ArrayAttr, mlir::ArrayAttr> getDPUTaskCoords(mlir::MLIRContext* ctx, ShapeRef shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "getDPUTaskCoords works with 4-d tensors only");
    Dim C = Dim(1), H = Dim(2), W = Dim(3);
    SmallVector<int32_t> start = {0, 0, 0};
    // subtract one due to the runtime specific
    SmallVector<int32_t> end = {static_cast<int32_t>(shape[W] - 1), static_cast<int32_t>(shape[H] - 1),
                                static_cast<int32_t>(shape[C] - 1)};
    auto startAttr = getInt32ArrayAttr(ctx, start);
    auto endAttr = getInt32ArrayAttr(ctx, end);
    return std::make_tuple(startAttr, endAttr);
}

static mlir::MemRefType buildMemSpaceHelper(mlir::MemRefType origType, mlir::Attribute memSpace) {
    return mlir::MemRefType::Builder(origType).setMemorySpace(memSpace);
}

static mlir::Value createHelperTensor(IERT::ConvolutionOp layer, mlir::OpBuilder& builder,
                                      const std::vector<int32_t>& weightTableVals,
                                      SmallVector<int64_t> weightTableShape) {
    auto ctx = layer.getContext();
    const DimsOrder dimsOrderZMajor = DimsOrder::NHWC;
    const auto weightTableType =
            mlir::MemRefType::get(weightTableShape, getSInt32Type(ctx), dimsOrderZMajor.toAffineMap(ctx));
    const auto weightTableStorageType = mlir::RankedTensorType::get(weightTableShape, getSInt32Type(ctx));

    const auto weightTableAttr = mlir::DenseElementsAttr::get(weightTableStorageType, makeArrayRef(weightTableVals));
    auto weightsTableConstOp = builder.create<IERT::ConstantOp>(layer->getLoc(), weightTableType, weightTableAttr);

    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    auto newTypeWeightsTableCMX = buildMemSpaceHelper(weightTableType, cmxMemSpaceAttr);

    auto weightsTableAllocOp =
            builder.create<mlir::memref::AllocOp>(weightsTableConstOp->getLoc(), newTypeWeightsTableCMX);
    auto copyOp = builder.create<IERT::CopyOp>(weightsTableConstOp->getLoc(), weightsTableConstOp.output(),
                                               weightsTableAllocOp);

    return copyOp.output();
}

static mlir::Value createWeightsTable(IERT::ConvolutionOp layer, mlir::OpBuilder& builder,
                                      const std::vector<int32_t>& weightTableVals) {
    auto outputs = layer.getOutputs();

    Dim C = Dim(1);
    const auto outputShape = getShape(outputs[0]);
    auto numChannelElements = outputShape[C];
    SmallVector<int64_t> weightTableShape{numChannelElements, 1, 1, 4};
    return createHelperTensor(layer, builder, weightTableVals, weightTableShape);
}

static mlir::Value createWeights(mlir::Operation* opLayer, mlir::OpBuilder& builder, const mlir::Value& filter) {
    if (opLayer == nullptr) {
        return mlir::Value();
    }

    auto layer = mlir::dyn_cast<LayerInterface>(opLayer);
    if (layer == nullptr) {
        return mlir::Value();
    }
    auto ctx = layer.getContext();

    const Dim Out(0);
    const Dim In(1);
    const Dim KernelY(2);
    const Dim KernelX(3);
    const DimsOrder deviceFilterOrder = DimsOrder::fromPermutation({Out, KernelY, KernelX, In});
    auto weightsType_nchw = filter.getType().dyn_cast<mlir::MemRefType>();
    auto weightsType_nhwc =
            mlir::MemRefType::get(weightsType_nchw.getShape(), weightsType_nchw.getElementType(),
                                  deviceFilterOrder.toAffineMap(ctx), weightsType_nchw.getMemorySpace());
    auto weights_nhwc_AllocOp = builder.create<mlir::memref::AllocOp>(opLayer->getLoc(), weightsType_nhwc);
    auto weights_nhwc_ddr = builder.create<IERT::ReorderOp>(opLayer->getLoc(), filter, weights_nhwc_AllocOp);

    auto weightsType = weights_nhwc_ddr.getType().dyn_cast<mlir::MemRefType>();
    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    auto newTypeWeightsCMX = buildMemSpaceHelper(weightsType, cmxMemSpaceAttr);

    auto weightsCMXAllocOp = builder.create<mlir::memref::AllocOp>(opLayer->getLoc(), newTypeWeightsCMX);
    auto weightsCMXAllocOpCopy = builder.create<IERT::CopyOp>(opLayer->getLoc(), weights_nhwc_ddr, weightsCMXAllocOp);

    return weightsCMXAllocOpCopy;
}

std::vector<int32_t> generateWTablesValues(const size_t channelSize, const int32_t weightPtrStep,
                                           const std::vector<int32_t>& biases) {
    std::vector<int32_t> weightsTableVals(channelSize * 4, 0);
    int32_t weightPtrOffset = 0;
    for (size_t i = 0; i < weightsTableVals.size(); i += 4) {
        const size_t PRELU_SCALE_OFFSET = 0;
        const int8_t PRELU_SCALE_VALUE = 1;
        // FIXME PPE shift is actually 6 bit long, 2 higher bits stand for rounding mode
        const size_t PPE_SHIFT_OFFSET = 8;
        const int8_t PPE_SHIFT_VALUE = 0;
        const size_t PPE_MULT_OFFSET = 16;
        // FIXME PPE multiplier has sign, which may affect lower bits
        const int16_t PPE_MULT_VALUE = 1;

        const int32_t mult_shift = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) + (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) +
                                   (PPE_MULT_VALUE << PPE_MULT_OFFSET);
        const int32_t PPE_BIAS_VALUE = biases.at(i / 4);

        weightsTableVals.at(i + 0) = weightPtrOffset;
        weightsTableVals.at(i + 1) = 0x0;
        weightsTableVals.at(i + 2) = mult_shift;
        weightsTableVals.at(i + 3) = PPE_BIAS_VALUE;

        weightPtrOffset += weightPtrStep;
    }

    return weightsTableVals;
}

//
// ConvRewrite
//

class ConvertToNCEOpsPass::ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, Logger log, IERT::RunTimeResourcesOp resources)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _log(log), _resources(resources) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mutable IERT::RunTimeResourcesOp _resources;
};

mlir::LogicalResult ConvertToNCEOpsPass::ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    size_t availableNNCMXSizeInBytes = _resources.getAvailableMemory(cmxMemSpaceAttr).byteSize();
    size_t requiredNNCMXBytes = 0;
    for (auto val : {origOp.filter(), origOp.input(), origOp.output()}) {
        auto type = val.getType().cast<mlir::MemRefType>();
        requiredNNCMXBytes += getTypeTotalSize(type).count();
    }
    static const size_t NUM_ELEMENTS_PER_WEIGHT_TABLE_ROW = 4;
    requiredNNCMXBytes += origOp.filter().getType().cast<mlir::ShapedType>().getShape().front() *
                          NUM_ELEMENTS_PER_WEIGHT_TABLE_ROW * 4;

    if (requiredNNCMXBytes > availableNNCMXSizeInBytes / 4) {
        return matchFailed(rewriter, origOp, "CMX memory is not enough");
    }

    for (const auto& operand : origOp.getOpOperands()) {
        auto shape = operand.get().getType().cast<mlir::ShapedType>().getShape();  // TODO: Fix this assumption.
        constexpr int CHANNEL_ALIGNMENT = 16;
        if (shape[1] % CHANNEL_ALIGNMENT != 0) {
            return matchFailed(rewriter, origOp, "Channels are not aligned");
        }
    }

    // prepare weights in CMX
    const auto dim_c = Dim(1);
    const auto dim_h = Dim(2);
    const auto dim_w = Dim(3);
    const auto filter_shape = getShape(origOp.filter());

    auto weightsCMXAllocOpCopy = createWeights(origOp, rewriter, origOp.filter());
    if (!weightsCMXAllocOpCopy) {
        return matchFailed(rewriter, origOp, "Failed to create weights");
    }

    auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    auto padsEnd = parseIntArrayAttr(origOp.pads_end());

    auto padsBeginAttr = getInt32ArrayAttr(ctx, padsBegin);
    auto padsEndAttr = getInt32ArrayAttr(ctx, padsEnd);

    auto kernelPaddingAttr = getInt32ArrayAttr(ctx, makeArrayRef({padsBegin[0], padsBegin[1], padsEnd[0], padsEnd[1]}));

    const auto outputShape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    std::vector<int32_t> biases(outputShape[dim_c]);

    if (origOp.bias()) {
        auto biasConst = origOp.bias().getDefiningOp<ConstantInterface>();
        for (auto p : enumerate(biasConst.getContent().getValues<float>())) {
            int32_t biasVal = std::lround(p.value() * 65536.f);  // FIXME 2 ^ 16 might be more obvious
            biases.at(p.index()) = biasVal;
        }
    }
    int32_t weightPtrStep = static_cast<int32_t>(filter_shape[dim_c]) * static_cast<int32_t>(filter_shape[dim_h]) *
                            static_cast<int32_t>(filter_shape[dim_w]) * sizeof(int16_t);
    std::vector<int32_t> weightTableVals =
            generateWTablesValues(static_cast<int32_t>(outputShape[dim_c]), weightPtrStep, biases);

    auto weightsTableAllocOpCopy = createWeightsTable(origOp, rewriter, weightTableVals);

    //                              memref (null) -> NCEOp -> memref (null)
    // memref (null) -> CopyOp -> memref (CMX_NN) -> NCEOp -> memref (CMX_NN) -> CopyOp -> memref (null)

    // NCHW -> NHWC for input
    const DimsOrder dimsOrderZMajor = DimsOrder::NHWC;

    auto inputType_nchw = origOp.input().getType().dyn_cast<mlir::MemRefType>();
    auto inputType_nhwc = mlir::MemRefType::get(inputType_nchw.getShape(), inputType_nchw.getElementType(),
                                                dimsOrderZMajor.toAffineMap(ctx), inputType_nchw.getMemorySpace());

    auto input_nhwc_AllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), inputType_nhwc);
    auto input_nhwc_ddr = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), origOp.input(), input_nhwc_AllocOp);

    auto inputType = input_nhwc_ddr.getType().dyn_cast<mlir::MemRefType>();
    // prepare input in CMX
    auto newTypeInputCMX = buildMemSpaceHelper(inputType, cmxMemSpaceAttr);
    auto inputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeInputCMX);
    auto inputCMXAllocOpCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), input_nhwc_ddr, inputCMXAllocOp);
    // end input processing

    // prepare output in CMX
    auto outputType = origOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto newTypeOutputCMX =
            mlir::MemRefType::get(outputType.getShape(), outputType.getElementType(), dimsOrderZMajor.toAffineMap(ctx),
                                  VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN));

    auto outputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeOutputCMX);

    auto ppeAttr = vpux::VPUIP::PPELayerTypeAttr::get(ctx, VPUIP::PPELayerType::NOOP);

    mlir::ArrayAttr startAttr, endAttr;
    const auto shape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    std::tie(startAttr, endAttr) = getDPUTaskCoords(ctx, shape);

    const auto filterShape = getShape(origOp.filter());

    static const auto KH = Dim(2);
    static const auto KW = Dim(3);
    const SmallVector<int64_t> start = {filterShape[KW], filterShape[KH]};
    mlir::ArrayAttr kernelSize = getInt32ArrayAttr(ctx, start);

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), inputCMXAllocOpCopy, weightsCMXAllocOpCopy,
                                                          weightsTableAllocOpCopy, nullptr, inputCMXAllocOpCopy,
                                                          outputCMXAllocOp, outputCMXAllocOp, VPUIP::NCETaskType::CONV,
                                                          ppeAttr, kernelPaddingAttr, origOp.strides(), kernelSize);

    nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);

    // CMX -> DDR
    auto cmx_nhwc_mem_ref = nceOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto ddr_nhwc_type = mlir::MemRefType::get(cmx_nhwc_mem_ref.getShape(), cmx_nhwc_mem_ref.getElementType(),
                                               cmx_nhwc_mem_ref.getAffineMaps(),
                                               origOp.output().getType().dyn_cast<mlir::MemRefType>().getMemorySpace());
    auto ddr_nhwc_alloc_op = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), ddr_nhwc_type);
    auto output_ddr_nhwc = rewriter.create<IERT::CopyOp>(origOp->getLoc(), nceOp.output(), ddr_nhwc_alloc_op);

    // NHWC -> NCHW
    auto outputReorderOp = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), output_ddr_nhwc, origOp.output_buff());

    rewriter.replaceOp(origOp, outputReorderOp->getResults());

    return mlir::success();
}

void ConvertToNCEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    if (!resources) {
        _log.error("Could not retrieve IERT.RunTimeResources");
        signalPassFailure();
    }

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, _log, resources);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}
