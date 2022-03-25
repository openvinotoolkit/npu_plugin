//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

#include <unordered_set>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const Shape kernelSize{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto strideW = kernelStrides[Dims4D::Strides::X];

    int64_t activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::DW_CONV, kernelSize, strideW,
                                                                        input.getElementType(), 1);

    Byte requiredCMX(0);

    for (const auto& type : {input, filter, output}) {
        requiredCMX += type.getTotalAllocSize();
    }

    requiredCMX += NCEInvariant::getWeightsTableSize(OC);
    requiredCMX += activationWindowSize * 1_Byte;

    return requiredCMX <= getTotalCMXSize(getOperation());
}

//
// isSupported
//

bool vpux::VPU::NCEDepthConvolutionOp::isSupported(IE::GroupConvolutionOp op, NCEInvariant::LogCb logCb) {
    if (op.getType().getRank() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }
    if (getShape(op.filter()).size() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(llvm::formatv("Dilated convolution is not supported"));
        return false;
    }

    const auto inputShape = getShape(op.input());
    const auto IC = inputShape[Dims4D::Act::C];

    const auto filterShape = getShape(op.filter());
    const auto fIC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!op.groups().hasValue()) {
        logCb(llvm::formatv("Grouped convolution does not have groups attribute"));
        return false;
    }
    if (op.groups().getValue() != OC) {
        logCb(llvm::formatv("Unsupported group size: '{0}' expected '{1}'", op.groups(), OC));
        return false;
    }
    if (fIC != 1) {
        logCb(llvm::formatv("Group Convolution with more than one filter per input channel is not supported"));
        return false;
    }
    if (OC != IC) {
        logCb(llvm::formatv("Group Convolution has '{0}' groups, expected '{1}'", OC, IC));
        return false;
    }

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.pads_begin(), op.pads_end());

    if (!NCEInvariant::isAttrsSupported(VPU::getArch(op), KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right,
                                        logCb)) {
        return false;
    }

    const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = op.filter().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();

    if (!NCEInvariant::isActTypeSupported(inputType, getInputChannelAlignmentImpl(inputType)) ||
        !NCEInvariant::isActTypeSupported(outputType, getOutputChannelAlignmentImpl(outputType))) {
        logCb(llvm::formatv("Misaligned tensor shape"));
        return false;
    }

    const auto inputOrder = inputType.getDimsOrder();
    const auto filterOrder = filterType.getDimsOrder();
    const auto outputOrder = outputType.getDimsOrder();

    if (inputOrder != DimsOrder::NHWC || filterOrder != DimsOrder::OYXI || outputOrder != DimsOrder::NHWC) {
        logCb(llvm::formatv("Unsupported layout"));
        return false;
    }

    return true;
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(NCEDepthConvolutionOp op) {
    const auto arch = getArch(op);

    const NCEConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions());
    if (mlir::failed(verifyConv(op->getLoc(), arch, convAdaptor, op.output()))) {
        return mlir::failure();
    }

    const auto inputOrder = DimsOrder::fromValue(op.input());
    const auto filterOrder = DimsOrder::fromValue(op.filter());
    const auto outputOrder = DimsOrder::fromValue(op.output());

    if (inputOrder != DimsOrder::NHWC || filterOrder != DimsOrder::OYXI || outputOrder != DimsOrder::NHWC) {
        return errorAt(op, "Unsupported operand/result layout, expected NHWC");
    }

    const auto inputType = op.input().getType().cast<NDTypeInterface>();
    const auto IC = inputType.getShape()[Dims4D::Act::C];

    const auto outputType = op.output().getType().cast<NDTypeInterface>();

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto kernelSize = Shape{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto activationWindowShape = getShape(op.activationWindow());
    const auto expectedActivationWindowShape = NCESparsity::inferActivationWindowShape(
            NCESparsity::Mode::DW_CONV, kernelSize, SX, inputType.getElementType(), 1);

    if (activationWindowShape != expectedActivationWindowShape) {
        return errorAt(op, "Got wrong shape for 'activationWindow' '{0}', expected '{1}'", activationWindowShape,
                       expectedActivationWindowShape);
    }

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize, SX,
                                                                    inputType.getElementType(), IC);

    if (op.activation_window_channel_length() != bitPatternSize) {
        return errorAt(op, "Got wrong value for 'activation_window_channel_length' '{0}', expected '{1}'",
                       op.activation_window_channel_length(), bitPatternSize);
    }

    const auto alignedFilterShape = getShape(op.filter());
    const auto expectedAlignedFilterShape = op.inferAlignedFilterShape(outputType);

    if (alignedFilterShape != expectedAlignedFilterShape) {
        return errorAt(op, "Got wrong shape for 'filter' '{0}', expected '{1}'", alignedFilterShape,
                       expectedAlignedFilterShape);
    }

    return mlir::success();
}

Shape vpux::VPU::NCEDepthConvolutionOp::inferAlignedFilterShape(NDTypeInterface output) {
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(this->rawFilterShape()));
    const auto KY = rawFilterShape[Dims4D::Filter::KY];
    const auto KX = rawFilterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto alignment = NCEInvariant::getAlignment(output.getElementType());

    const auto remainder = (KY * KX) % alignment;

    if (remainder == 0) {
        return Shape{OC, 1, KY, KX};
    }

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    return Shape{OC, KY * KX + padding, 1, 1};
}

//
// InferShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEDepthConvolutionOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()));
    const auto fIC = filterShape[Dims4D::Filter::IC];

    if (fIC != 1) {
        return errorAt(loc, "Non depthwize convolution case");
    }

    // Adjust input shape to reuse helpers for standard convolution
    auto inShape = getShape(op.input()).toValues();
    inShape[Dims4D::Act::C] = 1;

    const auto windowStrides = parseIntArrayAttr<int64_t>(op.strides());
    const auto windowDilations = ngraph::Strides({1, 1});

    const auto padTop = op.pad().top().getValue().getSExtValue();
    const auto padBottom = op.pad().bottom().getValue().getSExtValue();
    const auto padLeft = op.pad().left().getValue().getSExtValue();
    const auto padRight = op.pad().right().getValue().getSExtValue();

    const auto dataPaddingBelow = ngraph::CoordinateDiff({padTop, padLeft});
    const auto dataPaddingAbove = ngraph::CoordinateDiff({padBottom, padRight});

    const auto outputShapeNG = ngraph::infer_convolution_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
            dataPaddingBelow, dataPaddingAbove, ngraph::Shape(filterShape.begin(), filterShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), windowDilations);

    const auto outputShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                                 return checked_cast<int64_t>(val);
                                             }));

    const auto elemType = op.input().getType().cast<vpux::NDTypeInterface>().getElementType();

    inferredReturnShapes.emplace_back(outputShape, elemType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEDepthConvolutionOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.setInput(0, DimsOrder::NHWC);
    info.setInput(1, DimsOrder::OYXI);
    info.setOutput(0, DimsOrder::NHWC);
}

//
// AlignedChannelsOpInterface
//

bool vpux::VPU::NCEDepthConvolutionOp::checkChannelRestrictions(int64_t channels) {
    const auto arch = getArch(*this);

    if (arch == VPU::ArchKind::MTL) {
        // HW restrictions for channel number
        static const std::unordered_set<int64_t> availableChannels{16, 32, 64};
        return availableChannels.count(channels) != 0;
    }

    return true;
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEDepthConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling =
            backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(), origPadding);

    // Drop the bias tile
    inputTiling.tiles.pop_back();

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(filter()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    inputTiling.tiles.push_back(VPU::getWeightsTableTile(this, outputTile));
    inputTiling.tiles.push_back(VPU::getActivationWindowTile(this, outputTile));

    return inputTiling;
}

void vpux::VPU::NCEDepthConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);

    const auto inputType = input().getType().cast<NDTypeInterface>();
    const auto IC = inputTiling.tiles[0].shape[Dims4D::Act::C];

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto kernelSize = Shape{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize, SX,
                                                                    inputType.getElementType(), IC);

    activation_window_channel_lengthAttr(getIntAttr(getContext(), bitPatternSize));
}
