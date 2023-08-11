//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

#include <unordered_set>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEMaxPoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem) {
    // TODO: VPUX37XX hw doesn't require weights table and activation window for max/average pool ops
    const auto outputShape = output.getShape();
    const auto outputChannels = outputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(kernel_size()));

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto strideW = kernelStrides[Dims4D::Strides::X];

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(outputChannels)};

    auto arch = getArch(getOperation());
    if (arch == VPU::ArchKind::VPUX30XX) {
        const auto activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::POOL, kernelSize,
                                                                               strideW, input.getElementType(), 1);
        buffers.push_back(activationWindowSize * 1_Byte);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEMaxPoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEMaxPoolOp::isSupported(IE::MaxPoolOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    auto arch = VPU::getArch(op);

    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (op.rounding_type() != IE::RoundingType::FLOOR) {
        logCb(formatv("Unsupported rounding mode '{0}'", op.rounding_type()));
        return false;
    }

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(op.kernel_size()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    if (KY != KX && arch != VPU::ArchKind::VPUX37XX) {
        logCb(formatv("Asymmetric kernel is not supported"));
        return false;
    }

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.pads_begin(), op.pads_end());

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(arch, inputType, getInputChannelAlignmentImpl(inputType), false) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);

    const auto logCb = [op](const formatv_object_base& msg) {
        (void)errorAt(op, "{0}", msg.str());
    };

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(kernel_size()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = pad().top().getValue().getSExtValue();
    const auto padBottom = pad().bottom().getValue().getSExtValue();
    const auto padLeft = pad().left().getValue().getSExtValue();
    const auto padRight = pad().right().getValue().getSExtValue();

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto inputType = input().getType().cast<NDTypeInterface>();
    const auto IC = inputType.getShape()[Dims4D::Act::C];

    const auto outputType = output().getType().cast<NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];

    const auto weightsTableShape = getShape(weightsTable());
    const auto expectedWeightsTableShape = NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape != expectedWeightsTableShape) {
        return errorAt(op, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                       expectedWeightsTableShape);
    }

    const auto activationWindowShape = getShape(activationWindow());
    const auto expectedActivationWindowShape = NCESparsity::inferActivationWindowShape(
            NCESparsity::Mode::POOL, kernelSize, SX, inputType.getElementType(), 1);

    if (activationWindowShape != expectedActivationWindowShape) {
        return errorAt(op, "Got wrong shape for 'activationWindow' '{0}', expected '{1}'", activationWindowShape,
                       expectedActivationWindowShape);
    }

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize, SX,
                                                                    inputType.getElementType(), IC);

    if (activation_window_channel_length() != bitPatternSize) {
        return errorAt(op, "Got wrong value for 'activation_window_channel_length' '{0}', expected '{1}'",
                       activation_window_channel_length(), bitPatternSize);
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEMaxPoolOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.input());

    const auto windowShape = parseIntArrayAttr<int64_t>(op.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(op.strides());

    const auto padTop = op.pad().top().getValue().getSExtValue();
    const auto padBottom = op.pad().bottom().getValue().getSExtValue();
    const auto padLeft = op.pad().left().getValue().getSExtValue();
    const auto padRight = op.pad().right().getValue().getSExtValue();

    const auto dataPaddingBelow = ngraph::CoordinateDiff({padTop, padLeft});
    const auto dataPaddingAbove = ngraph::CoordinateDiff({padBottom, padRight});

    const auto outputShapeNG = ngraph::infer_batched_pooling_forward(
            EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()), dataPaddingBelow, dataPaddingAbove,
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), /*is_window_all_in_padding_allowed=*/true,
            /*ceil_mode=*/false);

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

void vpux::VPU::NCEMaxPoolOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEMaxPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());

    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, kernel_size(), strides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    inputTiling.tiles.push_back(VPU::getWeightsTableTile(this, outputTile));
    inputTiling.tiles.push_back(VPU::getActivationWindowTile(this, outputTile));

    return inputTiling;
}

void vpux::VPU::NCEMaxPoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);

    const auto inputType = input().getType().cast<NDTypeInterface>();
    const auto IC = inputTiling.tiles[0].shape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(kernel_size()));

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize, SX,
                                                                    inputType.getElementType(), IC);

    activation_window_channel_lengthAttr(getIntAttr(getContext(), bitPatternSize));
}

OutputTiling vpux::VPU::NCEMaxPoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEMaxPoolOp::getKernelSize() {
    return parseIntArrayAttr<int64_t>(kernel_size());
}

SmallVector<int64_t> vpux::VPU::NCEMaxPoolOp::getStrides() {
    return parseIntArrayAttr<int64_t>(strides());
}

vpux::VPU::PaddingAttr vpux::VPU::NCEMaxPoolOp::getPad() {
    return padAttr();
}

bool vpux::VPU::NCEMaxPoolOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEMaxPoolOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCEMaxPoolOp shouldn't have a serializer");
}

bool vpux::VPU::NCEMaxPoolOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEMaxPoolOp::sparsitySupport() {
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
