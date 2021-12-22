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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

using namespace vpux;

//
// Run-time info
//

double vpux::VPUIP::getMemoryDerateFactor(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryDerateAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryBandwidthAttrName(), attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}

double vpux::VPUIP::getProcessorFrequency(IE::ExecutorResourceOp res) {
    VPUX_THROW_UNLESS(res.getKind() != nullptr, "Got empty executor resource kind");

    auto attr = res->getAttr(VPU::getProcessorFrequencyAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Executor resource '{0}' has no '{1}' attribute", res.getKind(),
                      VPU::getProcessorFrequencyAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Executor resource '{0}' has wrong '{1}' attribute : '{2}'",
                      res.getKind(), VPU::getProcessorFrequencyAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

//
// DW Convolution utility
//

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, int64_t alignment) {
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, alignment, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto origFilterType = weightsConst.output().getType().cast<mlir::ShapedType>();
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, int64_t alignment) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto flatWeightType =
            changeDimsOrder(changeShape(origFilterType, flatWeightShape), DimsOrder::fromValue(origFilter));
    auto flatWeightsOp = builder.create<IERT::GenericReshapeOp>(loc, flatWeightType, origFilter);

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = changeDimsOrder(flatWeightType, DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(flatWeightsOp.output()).toAffineMap(ctx));
    auto flatWeightsNCHW = builder.create<IERT::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.output(),
                                                               nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto outAllocType = changeDimsOrder(
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()), DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, alignment, 1, 1};
    const auto padValues = std::vector<ngraph::float16>(OC * alignment, 0.f);
    const auto padType =
            changeDimsOrder(mlir::RankedTensorType::get(padShape, origFilterType.getElementType()), DimsOrder::NCHW);
    const auto padAttr = mlir::DenseElementsAttr::get(padType, makeArrayRef(padValues));
    const auto padContentAttr = Const::ContentAttr::get(padAttr);

    const auto padAllocType = mlir::MemRefType::get(padShape, origFilterType.getElementType());
    const auto padAllocTypeNHWC = changeDimsOrder(padAllocType, DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, outAllocType);

    const SmallVector<int64_t> filterOffsets = {0, 0, 0, 0};
    const auto filterOffsetsAttr = getIntArrayAttr(ctx, filterOffsets);
    const auto flatWeightShapeAttr = getIntArrayAttr(ctx, flatWeightShape);

    const SmallVector<int64_t> paddingOffsets = {0, flatWeightChannelsCount, 0, 0};
    const auto paddingOffsetsAttr = getIntArrayAttr(ctx, paddingOffsets);
    const auto padShapeAttr = getIntArrayAttr(ctx, padShape);

    auto subViewFilter = builder.create<IERT::SubViewOp>(loc, subViewAlloc, filterOffsetsAttr, flatWeightShapeAttr);
    auto subViewPadding = builder.create<IERT::SubViewOp>(loc, subViewAlloc, paddingOffsetsAttr, padShapeAttr);

    auto subViewFilterCopy = builder.create<IERT::CopyOp>(loc, flatWeightsNCHW.result(), subViewFilter);
    auto subViewPaddingCopy = builder.create<IERT::CopyOp>(loc, paddedTensor.output(), subViewPadding);

    auto concatViewOp = builder.create<IERT::ConcatViewOp>(
            loc, SmallVector<mlir::Value>{subViewFilterCopy.output(), subViewPaddingCopy.output()}, subViewAlloc);

    // Step 5: Permute the result to NHWC.
    auto outNHWCType = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));

    auto outOpNCHW = builder.create<IERT::PermuteCastOp>(loc, outNHWCType, concatViewOp.output(), nhwcAttr, nchwAttr);

    return outOpNCHW.result();
}

}  // namespace

mlir::Value vpux::VPUIP::alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                     mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    const int64_t alignment = depthwiseConvAlignment - remainder;
    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};
    mlir::Value alignedFilter;
    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        alignedFilter = getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, alignment);
    } else {
        alignedFilter = getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, alignment);
    }
    return alignedFilter;
}

//
// CM Convolution utility
//

mlir::Value vpux::VPUIP::alignChannelMajorWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                        const mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto channelMajorConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % channelMajorConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Channel Major convolution does not provide constant weights");

    const int64_t alignment = channelMajorConvAlignment - remainder;
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightShape = Shape{OC, 1, 1, filtersPerInChan * KY * KX};
    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, 0, 0, alignment});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + alignment};
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);

    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

bool vpux::VPUIP::isChannelMajorCompatibleOperation(mlir::Operation* origOp, DimsOrder inDimsOrder,
                                                    int64_t inputChannels, int64_t inputTensorWidth,
                                                    VPU::ArchKind archKind) {
    auto convOpIE = mlir::dyn_cast<IE::ConvolutionOp>(origOp);
    if (convOpIE != nullptr) {
        const auto inType = convOpIE.input().getType().cast<mlir::ShapedType>().getElementType();
        if (!inType.isa<mlir::quant::QuantizedType>()) {
            return false;
        }
    }

    auto convOpIERT = mlir::dyn_cast<IERT::ConvolutionOp>(origOp);
    if (convOpIERT != nullptr) {
        const auto inType = convOpIERT.input().getType().cast<mlir::ShapedType>().getElementType();
        if (!inType.isa<mlir::quant::QuantizedType>()) {
            return false;
        }
    }

    return ((archKind == VPU::ArchKind::KMB) && (inDimsOrder == DimsOrder::NCHW) &&
            (inputChannels == VPUIP::NCEInvariant::NCE_CHANNEL_MAJOR_CONV_REQUIRED_CHANNELS_LIMIT) &&
            (inputTensorWidth % VPUIP::NCEInvariant::NCE_CHANNEL_MAJOR_CONV_REQUIRED_WIDTH_ALIGNMENT == 0));
}
