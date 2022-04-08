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
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <ngraph/validation_util.hpp>

#include <unordered_set>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEAveragePoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                   getArch(getOperation()), {input.getTotalAllocSize(), output.getTotalAllocSize()}) <=
           getTotalCMXSize(getOperation());
}

//
// isSupported
//

bool vpux::VPU::NCEAveragePoolOp::isSupported(IE::AvgPoolOp op, LogCb logCb, bool checkLayout,
                                              bool checkChannelAlignment) {
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
        if (!NCEInvariant::checkLayouts(op->getOperands(), op->getResult(0), arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(NCEAveragePoolOp op) {
    const auto arch = getArch(op);

    const auto logCb = [op](const formatv_object_base& msg) {
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
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEAveragePoolOpAdaptor op(operands, attrs);
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

void vpux::VPU::NCEAveragePoolOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEAveragePoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());

    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, kernel_size(), strides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    return inputTiling;
}

void vpux::VPU::NCEAveragePoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);
}

OutputTiling vpux::VPU::NCEAveragePoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEAveragePoolOp::getKernelSize() {
    return parseIntArrayAttr<int64_t>(kernel_size());
}

SmallVector<int64_t> vpux::VPU::NCEAveragePoolOp::getStrides() {
    return parseIntArrayAttr<int64_t>(strides());
}

vpux::VPU::PaddingAttr vpux::VPU::NCEAveragePoolOp::getPad() {
    return padAttr();
}

bool vpux::VPU::NCEAveragePoolOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEAveragePoolOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCEAveragePoolOp shouldn't have a serializer");
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEAveragePoolOp::sparsitySupport() {
    using vpux::VPU::SparsitySupport;
    const auto arch = getArch(this->getOperation());
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        VPUX_THROW("NCEAveragePoolOp is not supported for {0}", arch);
    case VPU::ArchKind::VPUX37XX:
        return SparsitySupport::SPARSE_OUTPUTS;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
