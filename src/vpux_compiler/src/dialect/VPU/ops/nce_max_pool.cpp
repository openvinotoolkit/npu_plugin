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

bool vpux::VPU::NCEMaxPoolOp::fitIntoCMX(mlir::Operation* op, mlir::ArrayAttr kernel_size, mlir::ArrayAttr strides,
                                         vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    const auto arch = getArch(op);

    Byte requiredCMX(0);

    for (const auto& type : {input, output}) {
        requiredCMX += type.getTotalAllocSize();
    }

    // MTL hw doesn't require weights table and activation window for max/average pool ops
    if (arch != VPU::ArchKind::MTL) {
        const auto outputShape = output.getShape();
        const auto OC = outputShape[Dims4D::Act::C];

        const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(kernel_size));

        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(strides));
        const auto SX = kernelStrides[Dims4D::Strides::X];

        const auto activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::POOL, kernelSize, SX,
                                                                               input.getElementType(), OC);

        requiredCMX += NCEInvariant::getWeightsTableSize(OC);
        requiredCMX += activationWindowSize * 1_Byte;
    }

    return requiredCMX <= getTotalCMXSize(op);
}

//
// isSupported
//

bool vpux::VPU::NCEMaxPoolOp::isSupported(IE::MaxPoolOp op, NCEInvariant::LogCb logCb) {
    if (op.getType().getRank() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }

    if (op.rounding_type() != IE::RoundingType::FLOOR) {
        logCb(llvm::formatv("Unsupported rounding mode '{0}'", op.rounding_type()));
        return false;
    }

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(op.kernel_size()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    if (KY != KX) {
        logCb(llvm::formatv("Assymetric kernel is not supported"));
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

    if (!NCEInvariant::isActTypeSupported(inputType, getInputChannelAlignment(inputType)) ||
        !NCEInvariant::isActTypeSupported(outputType, getOutputChannelAlignment(outputType))) {
        logCb(llvm::formatv("Misaligned tensor shape"));
        return false;
    }

    const auto inputOrder = inputType.getDimsOrder();
    const auto outputOrder = outputType.getDimsOrder();

    if (inputOrder != DimsOrder::NHWC || outputOrder != DimsOrder::NHWC) {
        logCb(llvm::formatv("Unsupported layout"));
        return false;
    }

    if (!fitIntoCMX(op, op.kernel_size(), op.strides(), inputType, outputType)) {
        logCb(llvm::formatv("Operation doesn't fit into CMX memory"));
        return false;
    }

    return true;
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(NCEMaxPoolOp op) {
    const auto arch = getArch(op);

    const auto logCb = [op](const llvm::formatv_object_base& msg) {
        (void)errorAt(op, "{0}", msg.str());
    };

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(op.kernel_size()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

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

//
// InferShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

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
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()), dataPaddingBelow, dataPaddingAbove,
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), /*is_window_all_in_padding_allowed=*/true,
            /*ceil_mode=*/false);

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

void vpux::VPU::NCEMaxPoolOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}

//
// AlignedChannelsOpInterface
//

bool vpux::VPU::NCEMaxPoolOp::checkChannelRestrictions(int64_t channels) {
    const auto arch = getArch(*this);

    if (arch == VPU::ArchKind::MTL) {
        // HW restrictions for channel number
        static const std::unordered_set<int64_t> availableChannels{16, 32, 64};
        return availableChannels.count(channels) != 0;
    }

    return true;
}
