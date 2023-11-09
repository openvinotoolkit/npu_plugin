//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"

#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/validation_util.hpp>

#include <numeric>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEInterpolateOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = getShape(op.input());

    const auto windowStrides = SmallVector<int64_t>{1, 1};
    const auto windowDilations = ngraph::Strides({1, 1});

    const auto dataPaddingBelow = ngraph::CoordinateDiff({0, 0});
    const auto dataPaddingAbove = ngraph::CoordinateDiff({0, 0});

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()));

    const auto outputShapeNG = ngraph::infer_convolution_forward(
            EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
            dataPaddingBelow, dataPaddingAbove, ngraph::Shape(filterShape.begin(), filterShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), windowDilations);

    const auto outShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));

    auto inType = op.input().getType();

    // Output type is independent of input type - if the input type is sparse, the output type is not guaranteed to be
    // sparse as well
    if (auto sparseInputType = inType.cast<VPU::SparseTensorType>()) {
        inType = sparseInputType.getData();
    }

    auto outType = inType.cast<vpux::NDTypeInterface>().changeShape(Shape(outShape));

    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::verify() {
    auto sparseInput = input().getType().dyn_cast<VPU::SparseTensorType>();
    if (sparseInput == nullptr) {
        return mlir::failure();
    }

    auto seAttr = sparseInput.getSeAttr().dyn_cast_or_null<VPU::SEInterpolateAttr>();
    if (seAttr == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEInterpolateOp::getKernelSizeVal() {
    const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));

    const auto KY = kernelShape[Dims4D::Filter::KY];
    const auto KX = kernelShape[Dims4D::Filter::KX];

    return {KY, KX};
}

SmallVector<int64_t> vpux::VPU::NCEInterpolateOp::getStridesVal() {
    return {1, 1};
}

vpux::VPU::PaddingAttr vpux::VPU::NCEInterpolateOp::getPad() {
    return VPU::getPaddingAttr(getContext(), PadInfo(0, 0, 0, 0));
}

bool isNCEInterpolateSupported(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType,
                               IE::InterpolateAttr attr, Optional<mlir::ArrayAttr> scalesAttr, VPU::ArchKind arch,
                               bool checkLayout, bool checkChannelAlignment, vpux::LogCb logCb) {
    // TODO E#71403: remove dimension check
    auto dimOver8K = [](ShapeRef shape) {
        for (auto dim : shape) {
            if (dim > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                return true;
            }
        }
        return false;
    };
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    if (dimOver8K(inputShape) || dimOver8K(outputShape)) {
        logCb(formatv("Dimension sizes over 8192 are not supported. Input shape {0}, output shape {1}", inputShape,
                      outputShape));
        return false;
    }

    if (attr == nullptr) {
        logCb(formatv("Missing Interpolate configuration information"));
        return false;
    }

    // Antialias is not supported
    if (attr.getAntialias() != nullptr && attr.getAntialias().getValue() == true) {
        logCb(formatv("Antialias is not supported"));
        return false;
    }

    // Only 4D interpolates are supported and the interpolation axes must be H and/or W
    if (inputShape.size() != 4 || outputShape.size() != 4) {
        logCb(formatv("Only 4D data is supported. Got {0}D input, {1}D output", inputShape.size(), outputShape.size()));
        return false;
    }
    if (outputShape[Dims4D::Act::N] != inputShape[Dims4D::Act::N]) {
        logCb(formatv("Interpolation over axis {0} is not supported", Dims4D::Act::N.ind()));
        return false;
    }
    if (outputShape[Dims4D::Act::C] != inputShape[Dims4D::Act::C]) {
        logCb(formatv("Interpolation over axis {0} is not supported", Dims4D::Act::C.ind()));
        return false;
    }

    // Check for the supported modes
    SmallVector<IE::InterpolateMode> supportedModes = {IE::InterpolateMode::NEAREST, IE::InterpolateMode::LINEAR,
                                                       IE::InterpolateMode::LINEAR_ONNX};
    if (llvm::find(supportedModes, attr.getMode().getValue()) == supportedModes.end()) {
        logCb(formatv("Mode {0} is not supported", attr.getMode().getValue()));
        return false;
    }

    // For linear interpolate, only ASYMMETRIC coordinate transformations mode is supported
    if (attr.getMode().getValue() == IE::InterpolateMode::LINEAR ||
        attr.getMode().getValue() == IE::InterpolateMode::LINEAR_ONNX) {
        SmallVector<IE::InterpolateCoordMode> supportedCoordModes = {IE::InterpolateCoordMode::ASYMMETRIC};
        if (llvm::find(supportedCoordModes, attr.getCoordMode().getValue()) == supportedCoordModes.end()) {
            logCb(formatv("Coordinate transformation mode {0} is not supported", attr.getCoordMode().getValue()));
            return false;
        }
    }

    // TODO E#83681: Add support for ALIGN_CORNERS mode
    if (attr.getMode().getValue() == IE::InterpolateMode::NEAREST) {
        if (attr.getCoordMode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) {
            logCb(formatv("Coordinate transformation mode {0} is not yet supported", attr.getCoordMode().getValue()));
            return false;
        }
    }

    // Only interpolate ops without padding are supported
    auto hasNonZeroPads = [&](mlir::ArrayAttr padsAttr) -> bool {
        if (padsAttr == nullptr) {
            return false;
        }
        auto pads = parseIntArrayAttr<int64_t>(padsAttr);
        return llvm::any_of(pads, [](int64_t pad) {
            return pad != 0;
        });
    };
    if (hasNonZeroPads(attr.getPadsBegin()) || hasNonZeroPads(attr.getPadsEnd())) {
        logCb(formatv("Padding is not supported"));
        return false;
    }

    // Scale must be integers; for linear modes, they must be in range [1-11]
    SmallVector<double> scales;
    auto shapeCalcModeAttr = attr.getShapeCalcMode();
    if (shapeCalcModeAttr != nullptr && shapeCalcModeAttr.getValue() == IE::InterpolateCalcMode::SCALES &&
        scalesAttr.has_value()) {
        scales = parseFPArrayAttr<double>(scalesAttr.value());
    } else {
        scales = {static_cast<double>(outputShape[Dims4D::Act::H]) / static_cast<double>(inputShape[Dims4D::Act::H]),
                  static_cast<double>(outputShape[Dims4D::Act::W]) / static_cast<double>(inputShape[Dims4D::Act::W])};
    }
    for (auto scale : scales) {
        if (std::floor(scale) != scale) {
            logCb(formatv("Only integer scales are supported. Got scale {0}", scale));
            return false;
        }
        if (attr.getMode().getValue() == IE::InterpolateMode::LINEAR ||
            attr.getMode().getValue() == IE::InterpolateMode::LINEAR_ONNX) {
            if (!(scale >= 1.0 && scale <= 11.0)) {
                logCb(formatv("Only scales in range [1-11] are supported for linear modes. Got scale {0}", scale));
                return false;
            }
        }
    }

    if (checkChannelAlignment) {
        if (!VPU::NCEInvariant::isInputActTypeSupported(arch, inputType,
                                                        VPU::NCEInterpolateOp::getInputChannelAlignmentImpl(inputType),
                                                        /*supportsInputActCompression=*/false) ||
            !VPU::NCEInvariant::isOutputActTypeSupported(
                    outputType, VPU::NCEInterpolateOp::getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        if (!VPU::NCEInvariant::checkLayouts({inputType}, {outputType}, arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

bool VPU::NCEInterpolateOp::isSupported(IE::InterpolateOp op, vpux::LogCb logCb, bool checkLayout,
                                        bool checkChannelAlignment) {
    auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();
    return isNCEInterpolateSupported(inputType, outputType, op.attr(), op.scales_attr(), VPU::getArch(op),
                                     checkChannelAlignment, checkLayout, logCb);
}

bool VPU::NCEInterpolateOp::isSupported(VPU::InterpolateOp op, vpux::LogCb logCb, bool checkLayout,
                                        bool checkChannelAlignment) {
    auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();
    return isNCEInterpolateSupported(inputType, outputType, op.attr(), op.scales_attr(), VPU::getArch(op),
                                     checkChannelAlignment, checkLayout, logCb);
}

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(
            VPU::getArch(*this), inputType, getInputChannelAlignment(), /*supportsInputActCompression=*/false));
}

//
// TilingBuilderOpInterace
//

TilingInfo vpux::VPU::NCEInterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(rawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();
    const auto strides = getIntArrayAttr(getContext(), getStridesVal());
    const auto padding = VPU::toPadInfo(getPad());

    auto inputTiling = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, padding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(weights()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    inputTiling.tiles.push_back(VPU::getWeightsTableTile(this, outputTile));

    return inputTiling;
}

void vpux::VPU::NCEInterpolateOp::adjustAttrs(const vpux::TilingInfo&, const vpux::TileInfo& outputTile) {
    // Same as NCEConvolution, but without padding
    VPU::adjustRawFilterShape(this, outputTile);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEInterpolateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEInterpolateOp::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy) {
    // TODO E#71871: enable SplitOverHeight and HKSwitch once runtime supports this
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverKernel;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEInterpolateOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, vpux::VPU::PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    return VPU::getNCEExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                                    distributionMode, numTiles, numClusters, alignment, kernel, pad,
                                                    stride, uniformDistributedSegments);
}

//
// fitIntoCMX
//

bool vpux::VPU::NCEInterpolateOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCEInterpolateOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output, Byte reservedMem) {
    const auto OC = output.getShape()[Dims4D::Act::C];
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// SparseOpInterface
//

vpux::VPU::SparsitySupport vpux::VPU::NCEInterpolateOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto arch = getArch(getOperation());
    const auto outputType = output().getType().cast<vpux::NDTypeInterface>();

    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);

    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::VPUX30XX: {
        // Layout will always be NHWC for VPUX30XX.
        return (VPU::SparsitySupport::SPARSE_INPUTS | VPU::SparsitySupport::SPARSE_WEIGHTS) & excludeMode;
    }

    case VPU::ArchKind::VPUX37XX:
        return NCESparsity::FULLY_SUPPORTED_SPARSITY_MODE & excludeMode;

    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEInterpolateOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("NCEInterpolateOp::serialize not implemented!");
}
