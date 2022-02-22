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

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

//
// memSizes
//

SmallVector<Byte> VPU::NCEConvolutionOp::memSizes(mlir::ArrayAttr strides, vpux::NDTypeInterface input,
                                                  vpux::NDTypeInterface filter, vpux::NDTypeInterface output) {
    SmallVector<Byte> requiredCMX(5, Byte(0));  // {input, filter, output, weightsTable, activationWindow} in order

    requiredCMX[0] = input.getTotalAllocSize();
    requiredCMX[2] = output.getTotalAllocSize();

    const auto filterShape = filter.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto inOrder = input.getDimsOrder();
    requiredCMX[3] = NCEInvariant::getWeightsTableSize(OC);

    if (inOrder == DimsOrder::NHWC) {
        requiredCMX[1] = filter.getTotalAllocSize();
        //  requiredCMX[4] activationWindow size equals 0
    } else if (inOrder == DimsOrder::NCHW) {
        const auto alignedFilter = vpux::IE::getAlignedFilterType({input, filter, output});

        const auto kernelSize = Shape{KY, KX};

        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides));
        const auto SX = kernelStrides[Dims4D::Strides::X];

        const auto activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::CM_CONV, kernelSize,
                                                                               SX, input.getElementType(), IC);

        requiredCMX[1] = alignedFilter.getTotalAllocSize();
        requiredCMX[4] = activationWindowSize * 1_Byte;
    } else {
        VPUX_THROW("[{0}] Unsupported input layout '{1}'", this->getLoc(), inOrder);
    }

    return requiredCMX;
}

//
// fitIntoCMX
//

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output) {
    Byte requiredCMX(0);

    auto memList = memSizes(strides(), input, filter, output);
    for (auto memItem : memList) {
        requiredCMX += memItem;
    }

    return requiredCMX <= getTotalCMXSize(*this);
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEConvolutionOp origOp, const TileInfo& outTile) {
    const auto origBiasShape = origOp.bias().hasValue() ? origOp.bias().getValue().getShape() : ShapeRef();
    const auto origPadding = toPadInfo(origOp.pad());

    auto tileConf = vpux::backInferConvTile(outTile, getShape(origOp.input()), getShape(origOp.filter()), origBiasShape,
                                            origOp.strides(), origPadding);

    return {origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tileConf.tiles[0].offsets,
                                                                                    tileConf.tiles[0].shape),
            origOp.filter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tileConf.tiles[1].offsets,
                                                                                     tileConf.tiles[1].shape),
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape)};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPrefetch(VPU::NCEConvolutionOp origOp,
                                                                  vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    return {curTileTypes[0], vpux::IE::getAlignedFilterType(curTileTypes), curTileTypes[2],
            isWeightPrefetch ? vpux::IE::getAlignedFilterType(nextTileTypes) : nextTileTypes[0]};
}

//
// verifyPrefetchCMX
//

bool vpux::VPU::NCEConvolutionOp::verifyPrefetchCMX(const vpux::OutputTiling& tiling) {
    if (tiling.size() <= 1) {
        return false;
    }
    if (vpux::IE::isNestedTiling(tiling)) {
        return false;
    }

    Byte requiredCMX(0);
    auto requiredOperands = getRequiredOperandsForPrefetch(*this, tiling);
    VPUX_THROW_UNLESS(requiredOperands.size() == 4, "NCEConvolutionOp needs 4 operands for prefetch, got {0}",
                      requiredOperands.size());
    for (auto memItem : memSizes(strides(), requiredOperands[0], requiredOperands[1], requiredOperands[2])) {
        requiredCMX += memItem;
    }
    requiredCMX += requiredOperands[3].getTotalAllocSize();

    return requiredCMX <= getTotalCMXSize(*this);
}

//
// isSupported
//

bool vpux::VPU::NCEConvolutionOp::isSupported(IE::ConvolutionOp op, NCEInvariant::LogCb logCb) {
    if (op.getType().getRank() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(llvm::formatv("Dilated convolution is not supported"));
        return false;
    }

    const auto filterShape = getShape(op.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

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

    if (!NCEInvariant::isActTypeSupported(inputType, getInputChannelAlignment(inputType)) ||
        !NCEInvariant::isActTypeSupported(outputType, getOutputChannelAlignment(outputType))) {
        logCb(llvm::formatv("Misaligned tensor shape"));
        return false;
    }

    const auto arch = getArch(op);

    const auto inputOrder = inputType.getDimsOrder();
    const auto filterOrder = filterType.getDimsOrder();
    const auto outputOrder = outputType.getDimsOrder();

    if (inputOrder != DimsOrder::NHWC && inputOrder != DimsOrder::NCHW) {
        logCb(llvm::formatv("Unsupported input layout '{0}'", inputOrder));
        return false;
    }
    if (filterOrder != DimsOrder::OYXI) {
        logCb(llvm::formatv("Unsupported filter layout '{0}'", filterOrder));
        return false;
    }
    if (arch != VPU::ArchKind::MTL && outputOrder != DimsOrder::NHWC) {
        logCb(llvm::formatv("Unsupported output layout '{0}'", outputOrder));
        return false;
    }

    return true;
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyConv(mlir::Location loc, ArchKind arch, NCEConvolutionOpAdaptor op,
                                          mlir::Value output) {
    const auto logCb = [loc](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    if (op.bias() != nullptr) {
        const auto biasType = op.bias().getType();

        const auto outputShape = getShape(output);
        const auto OC = outputShape[Dims4D::Act::C];

        if (!biasType.getElementType().isa<mlir::FloatType>()) {
            return errorAt(loc, "Bias must have Float element type");
        }

        if (biasType.getRank() != 4 && biasType.getRank() != 1) {
            return errorAt(loc, "Bias must be either 1D or 4D");
        }

        if (biasType.getRank() == 1) {
            const auto numBiases = biasType.getNumElements();

            if (numBiases != OC) {
                return errorAt(loc, "Number of Biases values do not match output channels");
            }
        } else {
            const auto propBiasType = biasType.cast<vpux::NDTypeInterface>();
            const auto biasShape = propBiasType.getShape();

            if (biasShape[Dims4D::Act::N] != 1 || biasShape[Dims4D::Act::H] != 1 || biasShape[Dims4D::Act::W] != 1) {
                return errorAt(loc, "Biases must have 1 elements for N, H and W dimensions");
            }

            if (biasShape[Dims4D::Act::C] != OC) {
                return errorAt(loc, "Number of Biases channels do not match output channels");
            }
        }
    }

    const Shape filterShape = op.rawFilterShape() != nullptr ? Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()))
                                                             : getShape(op.filter()).toValues();
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = op.pad().top().getValue().getSExtValue();
    const auto padBottom = op.pad().bottom().getValue().getSExtValue();
    const auto padLeft = op.pad().left().getValue().getSExtValue();
    const auto padRight = op.pad().right().getValue().getSExtValue();

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::verifyOp(NCEConvolutionOp op) {
    const auto arch = getArch(op);
    return verifyConv(op->getLoc(), arch, op, op.output());
}

//
// InferShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEConvolutionOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.input());
    const Shape filterShape = op.rawFilterShape() != nullptr ? Shape(parseIntArrayAttr<int64_t>(op.rawFilterShape()))
                                                             : getShape(op.filter()).toValues();

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

void vpux::VPU::NCEConvolutionOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    const auto arch = VPU::getArch(*this);

    const auto canUseCMajor =
            NCEInvariant::isChannelMajorCompatible(arch, input().getType().cast<vpux::NDTypeInterface>());

    if (info.getInput(0) == DimsOrder::NCHW && canUseCMajor) {
        info.setInput(0, DimsOrder::NCHW);
    } else {
        info.setInput(0, DimsOrder::NHWC);
    }

    info.setInput(1, DimsOrder::OYXI);

    // FIXME: MTL ODU supports reordering of the output tensor, so we could use any layout here. But right now current
    // behavior of AdjustLayouts and OptimizeReorder passes might introduce extra Reorders in that case. We need to
    // update the passes to properly handle various Reorder propagation and fusing cases prior enabling ODU permutation
    // feature in MTL.
    info.setOutput(0, DimsOrder::NHWC);
}

//
// AlignedChannelsOpInterface
//

int64_t vpux::VPU::NCEConvolutionOp::getInputChannelAlignment(vpux::NDTypeInterface inputType) {
    const auto inOrder = inputType.getDimsOrder();
    if (inOrder == DimsOrder::NCHW) {
        // C-major convolution has no specific requirements
        return 1;
    }

    return NCEInvariant::getAlignment(inputType.getElementType());
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = getShape(filter());
    const auto origBiasShape = bias().hasValue() ? bias().getValue().getShape() : ShapeRef();
    const auto origPadding = toPadInfo(pad());

    return backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(), origPadding);
}

void vpux::VPU::NCEConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
    VPU::adjustBias(this, outputTile);
}
