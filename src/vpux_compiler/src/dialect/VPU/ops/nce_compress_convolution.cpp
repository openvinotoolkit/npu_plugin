//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCECompressConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                     vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCECompressConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                     vpux::NDTypeInterface output, Byte reservedMem) {
    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto inOrder = input.getDimsOrder();

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    VPUX_THROW_UNLESS(inOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCECompressConvolutionOp::isSupported(IE::ConvolutionOp op, LogCb logCb, bool checkLayout,
                                                      bool checkChannelAlignment) {
    const auto arch = getArch(op);

    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());

    const auto filterShape = getShape(op.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.pads_begin(), op.pads_end());

    const auto inputType = op.input().getType().cast<NDTypeInterface>();
    const auto filterType = op.filter().getType().cast<NDTypeInterface>();
    const auto outputType = op.output().getType().cast<NDTypeInterface>();

    return VPU::isNCEConvSupported(arch, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb);
}

//
// verifyOp
//

static mlir::LogicalResult verifyConv(mlir::Location loc, VPU::ArchKind arch, VPU::NCECompressConvolutionOpAdaptor op,
                                      mlir::Value output) {
    const auto logCb = [loc](const formatv_object_base& msg) {
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

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::verify() {
    auto op = getOperation();
    const auto arch = getArch(op);

    const NCECompressConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions());
    if (mlir::failed(verifyConv(getOperation()->getLoc(), arch, convAdaptor, output()))) {
        return mlir::failure();
    }

    const auto inputOrder = DimsOrder::fromValue(input());
    const auto filterOrder = DimsOrder::fromValue(filter());
    const auto outputOrder = DimsOrder::fromValue(output());

    VPUX_THROW_UNLESS(inputOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout [{1}], expected NHWC", getLoc(),
                      inputOrder);
    if (filterOrder != DimsOrder::OYXI) {
        return errorAt(op, "Unsupported 'filter' layout '{0}', expected OYXI", filterOrder);
    }
    if (arch != VPU::ArchKind::VPUX37XX && outputOrder != DimsOrder::NHWC) {
        return errorAt(op, "Unsupported 'output' layout '{0}', expected NHWC", outputOrder);
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCECompressConvolutionOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.input());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()));

    if (inShape[Dims4D::Act::C] != filterShape[Dims4D::Filter::IC]) {
        return errorAt(loc, "Input tensor channels and filter shape must be the same");
    }

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
    const auto outputType = inputType.cast<vpux::NDTypeInterface>().changeShape(Shape(outputShape));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCECompressConvolutionOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.setInput(0, DimsOrder::NHWC);
    info.setInput(1, DimsOrder::OYXI);
    // FIXME: VPUX37XX ODU supports reordering of the output tensor, so we could use any layout here. But right now
    // current behavior of AdjustLayouts and OptimizeReorder passes might introduce extra Reorders in that case. We need
    // to update the passes to properly handle various Reorder propagation and fusing cases prior enabling ODU
    // permutation feature in VPUX37XX.
    info.setOutput(0, DimsOrder::NHWC);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCECompressConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                         vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto origPadding = toPadInfo(pad());

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling =
            backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(), origPadding);
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

    return inputTiling;
}

void vpux::VPU::NCECompressConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
}

OutputTiling vpux::VPU::NCECompressConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

int64_t vpux::VPU::NCECompressConvolutionOp::getInputChannelAlignment() {
    return vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCECompressConvolutionOp::getKernelSize() {
    const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));
    const auto KY = kernelShape[Dims4D::Filter::KY];
    const auto KX = kernelShape[Dims4D::Filter::KX];
    return {KY, KX};
}

SmallVector<int64_t> vpux::VPU::NCECompressConvolutionOp::getStrides() {
    return parseIntArrayAttr<int64_t>(strides());
}

vpux::VPU::PaddingAttr vpux::VPU::NCECompressConvolutionOp::getPad() {
    return padAttr();
}

bool vpux::VPU::NCECompressConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
           strategy == VPU::MultiClusterStrategy::Clustering;
}

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::verifyChannels() {
    auto arch = VPU::getArch(*this);
    return mlir::success(
            vpux::VPU::NCEInvariant::isInputActTypeSupported(arch, input().getType().cast<vpux::NDTypeInterface>(),
                                                             getInputChannelAlignment(), true) &&
            vpux::VPU::NCEInvariant::isOutputActTypeSupported(output().getType().cast<vpux::NDTypeInterface>(),
                                                              getOutputChannelAlignment()));
}

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), true));
}

bool vpux::VPU::NCECompressConvolutionOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCECompressConvolutionOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCECompressConvolutionOp shouldn't have a serializer");
}
