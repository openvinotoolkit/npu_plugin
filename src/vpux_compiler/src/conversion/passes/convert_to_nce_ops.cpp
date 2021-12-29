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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/pwl_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Utilities
//

VPU::MPEMode getMpeForKmb(mlir::Type inElemType, mlir::Type outElemType, VPUIP::NCETaskType) {
    if (inElemType.dyn_cast<mlir::quant::QuantizedType>() != nullptr ||
        outElemType.dyn_cast<mlir::quant::QuantizedType>() != nullptr) {
        return VPU::MPEMode::MATRIX;
    }

    if (inElemType.isF16() || inElemType.isBF16() || outElemType.isF16() || outElemType.isBF16()) {
        return VPU::MPEMode::VECTOR_FP16;
    }

    // Let's fall back to vector (might be a bad idea though).
    return VPU::MPEMode::VECTOR;
}

VPU::MPEMode getMpeForMtl(mlir::Type, mlir::Type, VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        return VPU::MPEMode::CUBOID_16x16;
    case VPUIP::NCETaskType::DWCONV:
        return VPU::MPEMode::CUBOID_4x16;
    case VPUIP::NCETaskType::MAXPOOL:
        return VPU::MPEMode::CUBOID_4x16;
    case VPUIP::NCETaskType::ELTWISE:
        return VPU::MPEMode::CUBOID_8x16;
    default:
        return VPU::MPEMode::CUBOID_16x16;
    }
}

using MpeModeCb = VPU::MPEMode (*)(mlir::Type, mlir::Type, VPUIP::NCETaskType);
const EnumMap<VPU::ArchKind, MpeModeCb> mpeMap = {
        {VPU::ArchKind::KMB, getMpeForKmb},  //
        {VPU::ArchKind::TBH, getMpeForKmb},  //
        {VPU::ArchKind::MTL, getMpeForMtl},  //
};

mlir::Value retrieveMemrefOfCopyOp(mlir::Value val) {
    if (val != nullptr) {
        if (auto copyOp = val.getDefiningOp<IERT::CopyOp>()) {
            return copyOp.output_buff();
        }
    }
    return val;
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, int64_t OC, mlir::Value op_input,
                                     mlir::Value op_output, mlir::Value weights, mlir::Value bias,
                                     mlir::Value activationWindow) {
    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    // link weight table to memrefs rather than the operation of weights,
    // activation, and input for easy address pointer retrieval
    mlir::Value wMemref = retrieveMemrefOfCopyOp(weights);
    mlir::Value aMemref = retrieveMemrefOfCopyOp(activationWindow);
    mlir::Value iMemref = retrieveMemrefOfCopyOp(op_input);

    const auto dataType = mlir::MemRefType::get(weightTableShape, getSInt32Type(builder.getContext()));
    auto createWeightsTableOp =
            builder.create<VPUIP::WeightsTableOp>(loc, dataType, iMemref, op_output, wMemref, bias, aMemref);

    const auto dataTypeCMX = changeMemSpace(dataType, VPU::MemoryKind::CMX_NN);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, createWeightsTableOp.output(), dataAllocOp);

    return copyOp.output();
}

mlir::Value prepareTensorForDPU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) {
    // DMA DDR -> CMX
    const auto origType = input.getType().cast<mlir::MemRefType>();
    auto typeCMX = changeMemSpace(eraseTiledInfo(origType), VPU::MemoryKind::CMX_NN);
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, input, dmaAllocOp.memref());

    return dmaOp.output();
}

void addDPUTasks(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter, int64_t numDPU, int64_t opPadLeft,
                 int64_t opPadRight, int64_t opPadTop, int64_t opPadBottom, VPU::MPEMode mpeMode) {
    auto* ctx = nceOp.getContext();

    const auto outputShape = getShape(nceOp.output());
    const auto dpuTiles = VPUIP::DpuTiler::tileOverH(numDPU, outputShape, opPadLeft, opPadRight, opPadTop, opPadBottom);

    for (const auto& dpuTile : dpuTiles) {
        const auto startAttr = getIntArrayAttr(ctx, makeArrayRef(dpuTile.start));
        const auto endAttr = getIntArrayAttr(ctx, makeArrayRef(dpuTile.end));

        const auto pad = VPU::getPaddingAttr(ctx, dpuTile.padLeft, dpuTile.padRight, dpuTile.padTop, dpuTile.padBottom);

        nceOp.addDPUTask(rewriter, startAttr, endAttr, pad, mpeMode);
    }
}

struct QuantizationParams {
    SmallVector<int32_t> quantMult;
    SmallVector<int32_t> quantShift;
    int64_t postShift;
};

struct PostOpParams {
    VPU::PPEMode layerType;
    int64_t clampLow;
    int64_t clampHigh;
    int64_t LreluMult;
    int64_t LreluShift;
    Optional<QuantizationParams> quantParams;

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift) {
    }

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift,
                 const QuantizationParams& quantParams)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift),
              quantParams(quantParams) {
    }
};

int64_t getPwlClamp(VPUIP::NCEClusterTaskOp nceOp, const VPU::PPEMode ppeType, const bool getMin) {
    constexpr int64_t CLAMP_MIN = -4096;
    constexpr int64_t CLAMP_MAX = 4095;

    // Input type defines the compute type
    const auto inElemType = nceOp.input().getType().cast<mlir::MemRefType>().getElementType();
    if (inElemType.isa<mlir::FloatType>()) {
        return getMin ? CLAMP_MIN : CLAMP_MAX;
    }

    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    const auto rMin = quantReqs.input.rMin;
    const auto rMax = quantReqs.input.rMax;

    const auto outElemType = nceOp.output().getType().cast<mlir::MemRefType>().getElementType();
    if (const auto qElemType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto scale = qElemType.getScale();
        const auto actualMin = std::max(CLAMP_MIN, static_cast<int64_t>(std::ceil(rMin / scale)));
        const auto actualMax = std::min(CLAMP_MAX, static_cast<int64_t>(std::floor(rMax / scale)));
        return getMin ? actualMin : actualMax;
    }
    VPUX_THROW("Unexpected output element type: {0}", outElemType);
}

int64_t getPwlPostShift(const VPU::PPEMode ppeType) {
    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    return quantReqs.input.postShift;
}

PostOpParams getPwlPostOpParams(VPUIP::NCEClusterTaskOp nceOp, VPU::PPEMode ppeType) {
    const int64_t clampLow = getPwlClamp(nceOp, ppeType, true);
    const int64_t clampHigh = getPwlClamp(nceOp, ppeType, false);
    const int64_t LreluMult = 1;
    const int64_t LreluShift = 0;
    const int64_t postShift = getPwlPostShift(ppeType);

    // Dummy values for mult & shift, as the actual values will be computed in the weights table
    SmallVector<int32_t> quantMult = {1};
    SmallVector<int32_t> quantShift = {0};

    return PostOpParams{ppeType,   clampLow,   clampHigh,
                        LreluMult, LreluShift, QuantizationParams{quantMult, quantShift, postShift}};
}

std::pair<int64_t, int64_t> getClampValuesForQuantizedOps(mlir::quant::QuantizedType outElemQType,
                                                          mlir::Type outElemType) {
    const auto zps = extractScalesAndZeroPoints(outElemType).second;
    auto clampLow = outElemQType.getStorageTypeMin() - zps.front();
    auto clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    return {clampLow, clampHigh};
}

static mlir::Optional<PostOpParams> parsePostOp(VPUIP::NCEClusterTaskOp nceOp, IE::PostOp postOp,
                                                mlir::MemRefType origOutType, VPU::ArchKind arch) {
    if (postOp == nullptr) {
        return mlir::None;
    }

    auto outElemType = origOutType.getElementType();
    auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>();
    int64_t clampLowQuantized = 0;
    int64_t clampHighQuantized = 0;
    if (outElemQType != nullptr) {
        clampLowQuantized = getClampValuesForQuantizedOps(outElemQType, outElemType).first;
        clampHighQuantized = getClampValuesForQuantizedOps(outElemQType, outElemType).second;
    }

    if (postOp.name().getValue() == IE::ReLUOp::getOperationName()) {
        VPUX_THROW_UNLESS(postOp.attrs().empty(), "'{0}' PostOp should not have any attributes", postOp.name());

        int64_t clampLow = 0;
        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::LRELU, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::ClampOp::getOperationName()) {
        IE::ClampOp::Adaptor clamp(None, postOp.attrs());
        VPUX_THROW_UNLESS(clamp.verify(nceOp->getLoc()).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp",
                          postOp.attrs(), postOp.name());

        int64_t clampLow =
                (outElemQType != nullptr) ? clampLowQuantized : vpux::toFixedPoint(clamp.min().getValueAsDouble());
        int64_t clampHigh =
                (outElemQType != nullptr) ? clampHighQuantized : vpux::toFixedPoint(clamp.max().getValueAsDouble());
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::NOOP, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::LeakyReluOp::getOperationName()) {
        IE::LeakyReluOp::Adaptor leakyRelu(None, postOp.attrs());
        VPUX_THROW_UNLESS(leakyRelu.verify(nceOp->getLoc()).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp",
                          postOp.attrs(), postOp.name());

        int64_t clampLow = 0;
        if (outElemQType != nullptr) {
            clampLow =
                    (arch == VPU::ArchKind::MTL)
                            ? clampLowQuantized
                            : static_cast<int64_t>(clampLowQuantized / leakyRelu.negative_slope().getValueAsDouble());
        }
        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::LPRELU, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::SigmoidOp::getOperationName()) {
        return getPwlPostOpParams(nceOp, VPU::PPEMode::SIGMOID);
    } else if (postOp.name().getValue() == IE::TanhOp::getOperationName()) {
        return getPwlPostOpParams(nceOp, VPU::PPEMode::TANH);
    }

    VPUX_THROW("Unsupported PostOp '{0}'", postOp.name());
}

//
// ConvRewrite
//

class ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());
    auto filterDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.filter());

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX = changeMemSpace(eraseTiledInfo(outReorderType), VPU::MemoryKind::CMX_NN);

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, inputDPU, outAllocOpCMX.memref(),
                                                 filterDPU, origOp.bias(), nullptr);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr = VPU::getPaddingAttr(getContext(), padsBegin, padsEnd);

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, filterDPU, weightsTable, /*activation_window=*/nullptr,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::CONV, kernelSizeAttr, origOp.strides(),
            kernelPaddingAttr, /*activation_window_channel_length=*/nullptr, /*is_continued*/ nullptr);

    const auto mpeByType = mpeMap.at(_arch);
    const auto inElemType = origOp.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outElemType = origOutType.getElementType();
    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0],
                mpeByType(inElemType, outElemType, VPUIP::NCETaskType::CONV));
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr(), origOutType, _arch);
    if (postOpParams.hasValue()) {
        if (postOpParams->quantParams.hasValue()) {
            const auto quantParams = postOpParams->quantParams.getValue();
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift, quantParams.quantMult,
                             quantParams.quantShift, quantParams.postShift);
        } else {
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift);
        }
    }

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// MaxPoolRewrite
//

class MaxPoolRewrite final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
public:
    MaxPoolRewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    const auto elemType = getUInt8Type(builder.getContext());

    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    const auto dataType = mlir::MemRefType::get(fakeSparsityShape, elemType);
    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataType, Const::ContentAttr::get(dataAttr));

    const auto dataTypeCMX = changeMemSpace(dataType, VPU::MemoryKind::CMX_NN);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

mlir::LogicalResult MaxPoolRewrite::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);

    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_size()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType());

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());

    //
    // Generate activation window
    //

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX = changeMemSpace(eraseTiledInfo(outReorderType), VPU::MemoryKind::CMX_NN);

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), IC, inputDPU, outAllocOpCMX.memref(),
                                                 nullptr, nullptr, activationWindow);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr = VPU::getPaddingAttr(getContext(), padsBegin, padsEnd);

    const auto activation_window_channel_length = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, /*weights=*/nullptr, weightsTable, activationWindow,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(),
            kernelPaddingAttr, activation_window_channel_length, /*is_continued*/ nullptr);

    const auto mpeByType = mpeMap.at(_arch);
    const auto inElemType = origOp.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outElemType = origOutType.getElementType();
    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0],
                mpeByType(inElemType, outElemType, VPUIP::NCETaskType::MAXPOOL));
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr(), origOutType, _arch);
    if (postOpParams.hasValue()) {
        if (postOpParams->quantParams.hasValue()) {
            const auto quantParams = postOpParams->quantParams.getValue();
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift, quantParams.quantMult,
                             quantParams.quantShift, quantParams.postShift);
        } else {
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift);
        }
    }

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// GenericEltwiseConverter
//

template <class ConcreteOp>
class GenericEltwiseConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericEltwiseConverter<ConcreteOp>(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch,
                                        VPU::PPEMode ppeType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _numDPU(numDPU), _arch(arch), _ppeType(ppeType), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    VPU::PPEMode _ppeType;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericEltwiseConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Prepare input for DPU
    //

    auto firstInputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input1());
    auto secondInputDPU = origOp.input1() == origOp.input2()
                                  ? firstInputDPU
                                  : prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input2());

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().template cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX = changeMemSpace(eraseTiledInfo(outReorderType), VPU::MemoryKind::CMX_NN);

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(this->getContext(), static_cast<int32_t>(0));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), firstInputDPU, secondInputDPU,
                                                          /*weightsTable=*/nullptr,
                                                          /*activation_window=*/nullptr,
                                                          /*parent_input=*/firstInputDPU,
                                                          /*parent_output=*/outAllocOpCMX.memref(),
                                                          /*output_buff=*/outAllocOpCMX.memref(),
                                                          VPUIP::NCETaskType::ELTWISE,
                                                          /*kernel_size=*/nullptr,
                                                          /*kernel_strides=*/nullptr,
                                                          /*kernel_padding=*/nullptr, activation_window_channel_length,
                                                          /*is_continued*/ nullptr);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    auto outElemType = origOutType.getElementType();
    if (auto outElemQType = outElemType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr(), origOutType, _arch);
    if (postOpParams.hasValue()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization

    auto quantScale =
            VPU::calculateQuantScaleVectorForEltwise(origOp.input1().getType().template cast<mlir::ShapedType>(),
                                                     origOp.input2().getType().template cast<mlir::ShapedType>(),
                                                     origOutType, _arch, mlir::isa<IERT::MultiplyOp>(origOp));
    if (quantScale.hasValue()) {
        const auto scale = quantScale.getValue();

        const auto mult = getQuantMultFromScale(scale);
        const auto shifts = getQuantShiftAndPostShiftFromScale(scale);

        const auto shift = shifts.first;
        const auto post_shift = shifts.second;

        nceOp.addPPETask(rewriter, _ppeType, clampLow, clampHigh, LreluMult, LreluShift, SmallVector<int32_t>{mult},
                         SmallVector<int32_t>{shift}, post_shift);
    } else {
        nceOp.addPPETask(rewriter, _ppeType, clampLow, clampHigh, LreluMult, LreluShift);
    }

    //
    // Create DPU sub-task
    //

    const SmallVector<int64_t> padsBegin = {0, 0};
    const SmallVector<int64_t> padsEnd = {0, 0};
    const auto mpeByType = mpeMap.at(_arch);
    const auto inElemType = origOp.input1().getType().template cast<mlir::MemRefType>().getElementType();
    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0],
                mpeByType(inElemType, outElemType, VPUIP::NCETaskType::ELTWISE));

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// DepthwiseConvRewrite
//

class DepthwiseConvRewrite final : public mlir::OpRewritePattern<IERT::GroupConvolutionOp> {
public:
    DepthwiseConvRewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::GroupConvolutionOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult DepthwiseConvRewrite::matchAndRewrite(IERT::GroupConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());

    auto alignedFilter = VPUIP::alignDepthWiseWeightsTensor(rewriter, origOp->getLoc(), origOp.filter());
    auto filterDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), alignedFilter);

    //
    // Generate activation window
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto kernelSize = Shape{KY, KX};
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType());
    const auto actWindowChanLen = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), OC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX = changeMemSpace(eraseTiledInfo(outReorderType), VPU::MemoryKind::CMX_NN);

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, inputDPU, outAllocOpCMX.memref(),
                                                 filterDPU, origOp.bias(), activationWindow);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr = VPU::getPaddingAttr(getContext(), padsBegin, padsEnd);

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, filterDPU, weightsTable, activationWindow,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::DWCONV, kernelSizeAttr, origOp.strides(),
            kernelPaddingAttr, actWindowChanLen, /*is_continued*/ nullptr);

    const auto mpeByType = mpeMap.at(_arch);
    const auto inElemType = origOp.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outElemType = origOutType.getElementType();
    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0],
                mpeByType(inElemType, outElemType, VPUIP::NCETaskType::DWCONV));
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr(), origOutType, _arch);
    if (postOpParams.hasValue()) {
        if (postOpParams->quantParams.hasValue()) {
            const auto quantParams = postOpParams->quantParams.getValue();
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift, quantParams.quantMult,
                             quantParams.quantShift, quantParams.postShift);
        } else {
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift);
        }
    }

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// ConvertToNCEOpsPass
//

class ConvertToNCEOpsPass final : public ConvertToNCEOpsBase<ConvertToNCEOpsPass> {
public:
    explicit ConvertToNCEOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertToNCEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    VPUX_THROW_UNLESS(mpeMap.find(arch) != mpeMap.end(), "Failed to map MPE mode to target arch");

    auto nceCluster = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(VPU::ExecutorKind::DPU);
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<MaxPoolRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<GenericEltwiseConverter<IERT::AddOp>>(&ctx, dpuExec.count(), arch, VPU::PPEMode::ADD, _log);
    patterns.insert<GenericEltwiseConverter<IERT::MultiplyOp>>(&ctx, dpuExec.count(), arch, VPU::PPEMode::MULT, _log);
    patterns.insert<GenericEltwiseConverter<IERT::SubtractOp>>(&ctx, dpuExec.count(), arch, VPU::PPEMode::SUB, _log);
    patterns.insert<GenericEltwiseConverter<IERT::AndOp>>(&ctx, dpuExec.count(), arch, VPU::PPEMode::AND, _log);
    patterns.insert<DepthwiseConvRewrite>(&ctx, dpuExec.count(), arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}
