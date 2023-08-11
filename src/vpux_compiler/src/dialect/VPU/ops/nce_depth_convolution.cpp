//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <ngraph/validation_util.hpp>

#include <unordered_set>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output, Byte reservedMem) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const Shape kernelSize{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto strideW = kernelStrides[Dims4D::Strides::X];

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    auto arch = getArch(getOperation());
    if (arch == VPU::ArchKind::VPUX30XX) {
        int64_t activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::DW_CONV, kernelSize,
                                                                            strideW, input.getElementType(), 1);
        buffers.push_back(activationWindowSize * 1_Byte);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEDepthConvolutionOp::isSupported(IE::GroupConvolutionOp op, LogCb logCb, bool checkLayout,
                                                   bool checkChannelAlignment) {
    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }
    if (getShape(op.filter()).size() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
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
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return false;
    }
    if (op.groups().getValue() != OC) {
        logCb(formatv("Unsupported group size: '{0}' expected '{1}'", op.groups(), OC));
        return false;
    }
    if (fIC != 1) {
        logCb(formatv("Group Convolution with more than one filter per input channel is not supported"));
        return false;
    }
    if (OC != IC) {
        logCb(formatv("Group Convolution has '{0}' groups, expected '{1}'", OC, IC));
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
    const auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(getArch(op), inputType, getInputChannelAlignmentImpl(inputType),
                                                   false) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto arch = getArch(op);
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 2, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult verifyDepthConv(mlir::Location loc, VPU::ArchKind arch, VPU::NCEDepthConvolutionOpAdaptor op,
                                    mlir::Value output) {
    const auto logCb = [loc](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);
    const auto OC = outputShape[Dims4D::Act::C];

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = op.pad().top().getValue().getSExtValue();
    const auto padBottom = op.pad().bottom().getValue().getSExtValue();
    const auto padLeft = op.pad().left().getValue().getSExtValue();
    const auto padRight = op.pad().right().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto weightsTableShape = getShape(op.weightsTable());
    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                       expectedWeightsTableShape);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);

    const NCEDepthConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions());
    if (mlir::failed(verifyDepthConv(op->getLoc(), arch, convAdaptor, output()))) {
        return mlir::failure();
    }

    const auto loc = op->getLoc();
    const auto logCb = [loc](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };
    if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 2, logCb)) {
        return errorAt(op, "Unsupported layouts");
    }

    const auto inputType = input().getType().cast<NDTypeInterface>();
    const auto IC = inputType.getShape()[Dims4D::Act::C];

    const auto outputType = output().getType().cast<NDTypeInterface>();

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto kernelSize = Shape{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto activationWindowShape = getShape(activationWindow());
    const auto expectedActivationWindowShape = NCESparsity::inferActivationWindowShape(
            NCESparsity::Mode::DW_CONV, kernelSize, SX, inputType.getElementType(), 1);

    if (activationWindowShape != expectedActivationWindowShape) {
        return errorAt(op, "Got wrong shape for 'activationWindow' '{0}', expected '{1}'", activationWindowShape,
                       expectedActivationWindowShape);
    }

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize, SX,
                                                                    inputType.getElementType(), IC);

    if (activation_window_channel_length() != bitPatternSize) {
        return errorAt(op, "Got wrong value for 'activation_window_channel_length' '{0}', expected '{1}'",
                       activation_window_channel_length(), bitPatternSize);
    }

    const auto alignedFilterShape = getShape(filter());
    const auto expectedAlignedFilterShape = inferAlignedFilterShape(outputType);

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
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

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
            EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
            dataPaddingBelow, dataPaddingAbove, ngraph::Shape(filterShape.begin(), filterShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), windowDilations);

    const auto outputShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                                 return checked_cast<int64_t>(val);
                                             }));

    auto inputType = op.input().getType();
    if (auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        inputType = sparseInputType.getData();
    }
    const auto outputType = inputType.cast<vpux::NDTypeInterface>().changeShape(Shape(outputShape));

    inferredReturnTypes.push_back(outputType);
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
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEDepthConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                      vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling =
            backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    // Remove bias input tile if present
    if (inputTiling.tiles.size() > 2) {
        // Drop the bias tile
        inputTiling.tiles.pop_back();
    }

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

OutputTiling vpux::VPU::NCEDepthConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEDepthConvolutionOp::getKernelSize() {
    const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = kernelShape[Dims4D::Filter::KY];
    const auto KX = kernelShape[Dims4D::Filter::KX];
    return {KY, KX};
}

SmallVector<int64_t> vpux::VPU::NCEDepthConvolutionOp::getStrides() {
    return parseIntArrayAttr<int64_t>(strides());
}

vpux::VPU::PaddingAttr vpux::VPU::NCEDepthConvolutionOp::getPad() {
    return padAttr();
}

bool vpux::VPU::NCEDepthConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

bool vpux::VPU::NCEDepthConvolutionOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEDepthConvolutionOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCEDepthConvolutionOp shouldn't have a serializer");
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEDepthConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto arch = getArch(getOperation());
    const auto outputType = output().getType().cast<vpux::NDTypeInterface>();
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return VPU::SparsitySupport::NONE;
    case VPU::ArchKind::VPUX37XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
