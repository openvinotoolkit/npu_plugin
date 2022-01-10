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
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(mlir::Operation* op, mlir::ShapedType input, mlir::ShapedType filter,
                                             mlir::ShapedType output) {
    const auto outputShape = getShape(output);
    const auto OC = outputShape[Dims4D::Act::C];

    Byte requiredCMX(0);

    for (const auto& type : {input, filter, output}) {
        requiredCMX += getTotalSize(type);
    }

    requiredCMX += NCEInvariant::getWeightsTableSize(OC);

    return requiredCMX <= getTotalCMXSize(op);
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
            const auto biasShape = getShape(biasType);

            if (biasShape[Dims4D::Act::N] != 1 || biasShape[Dims4D::Act::H] != 1 || biasShape[Dims4D::Act::W] != 1) {
                return errorAt(loc, "Biases must have 1 elements for N, H and W dimensions");
            }

            if (biasShape[Dims4D::Act::C] != OC) {
                return errorAt(loc, "Number of Biases channels do not match output channels");
            }
        }
    }

    const auto filterShape = getShape(op.filter());
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
    const auto filterShape = getShape(op.filter());

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

    const auto elemType = op.input().getType().cast<mlir::ShapedType>().getElementType();

    inferredReturnShapes.emplace_back(outputShape, elemType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEConvolutionOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.setInput(0, DimsOrder::NHWC);
    info.setInput(1, DimsOrder::OYXI);
    info.setOutput(0, DimsOrder::NHWC);
}
