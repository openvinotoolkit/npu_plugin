//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
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

namespace {
bool checkPadding(ArrayRef<int64_t> pads) {
    if (pads.size() != 4) {
        return false;
    }

    // There's a way to calculate this by searching the non-zero padding.
    // But let's keep it simple.
    return pads[Dims4D::Act::N.ind()] == 0 && pads[Dims4D::Act::C.ind()] >= 0 && pads[Dims4D::Act::H.ind()] == 0 &&
           pads[Dims4D::Act::W.ind()] == 0;
}

bool checkQuantization(const mlir::Type outputType) {
    const auto ndType = outputType.cast<vpux::NDTypeInterface>();
    if (ndType == nullptr) {
        return false;
    }
    const auto elemType = ndType.getElementType();
    return elemType.isa<mlir::quant::UniformQuantizedType>() || elemType.isF16() || elemType.isF32();
}

std::vector<int64_t> expandShape(const ShapeRef shape, const VPU::PaddingAttr pads) {
    const auto& top = pads.top().getInt();
    const auto& bottom = pads.bottom().getInt();
    const auto& left = pads.left().getInt();
    const auto& right = pads.right().getInt();

    const std::vector<int64_t> targetShape = {shape[Dims4D::Act::N], shape[Dims4D::Act::C],
                                              shape[Dims4D::Act::H] + top + bottom,
                                              shape[Dims4D::Act::W] + left + right};

    return targetShape;
}

bool isSuperdenseRequired(const DimsOrder inputOrder, const ShapeRef outputShape, const DimsOrder outputOrder,
                          const mlir::Type outElemType) {
    // When input order matches output order, permutation is not required.
    // Therefore, super-dense mode is not required for the operation.
    if (inputOrder == outputOrder) {
        return false;
    }

    // If the inner-most dimension of output shape is aligned, super-dense mode is not required.
    const auto outputMemShape = outputOrder.toMemoryOrder(outputShape);
    const auto outputInnerDim = outputMemShape.back();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto outputInnerDimRemainder = outputInnerDim % alignment;
    return outputInnerDimRemainder != 0;
}
}  // namespace

//
// fitIntoCMX
//

bool vpux::VPU::NCEPermuteQuantizeOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                   getArch(getOperation()), {input.getTotalAllocSize(), output.getTotalAllocSize()}) <=
           getTotalCMXSize(getOperation());
}

//
// isSupported
//

bool vpux::VPU::NCEPermuteQuantizeOp::isSupported(IE::PermuteQuantizeOp op, LogCb logCb) {
    // First of all, this operation makes sense only when ODU permutation is supported.
    const auto arch = getArch(op);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        logCb(formatv("Target architecture {0} does not support ODU permutations", arch));
        return false;
    }

    // Check that quantization is per-tensor.
    if (!checkQuantization(op.output().getType())) {
        logCb(formatv("Only per-tensor quantization is supported. Got: {0}", op.output().getType()));
        return false;
    }

    // Check padding. NCEPermuteQuantizeOp supports only padding over channels.
    const auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_beginAttr());
    if (padsBegin.size() != 4) {
        logCb(formatv("Only four-dimensional start paddings are supported. Got: {0}", padsBegin));
        return false;
    }
    if (!checkPadding(padsBegin)) {
        logCb(formatv("Only start padding of channels is supported. Got: {0}", padsBegin));
        return false;
    }
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_endAttr());
    if (padsEnd.size() != 4) {
        logCb(formatv("Only four-dimensional end paddings are supported. Got: {0}", padsBegin));
        return false;
    }
    if (!checkPadding(padsEnd)) {
        logCb(formatv("Only end padding of channels is supported. Got: {0}", padsEnd));
        return false;
    }

    const auto inputOrder = DimsOrder::fromValue(op.input());
    const auto expectedInOrder = DimsOrder::NCHW;
    if (inputOrder != expectedInOrder) {
        logCb(formatv("Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inputOrder));
        return false;
    }

    const auto outputOrder = DimsOrder::fromValue(op.output());
    const auto expectedOutOrder = DimsOrder::NHWC;
    if (outputOrder != expectedOutOrder) {
        logCb(formatv("Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outputOrder));
        return false;
    }

    const auto outElemType = op.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto inputShape = getShape(op.input());
    if (!IE::isShapeCompatibleWithODUPermute(inputShape, alignment)) {
        logCb(formatv("Cannot cast input shape {0} to 1xWxCxH", inputShape));
        return false;
    }

    const auto outputShape = getShape(op.output());
    if (!IE::isShapeCompatibleWithODUPermute(outputShape, alignment)) {
        logCb(formatv("Cannot cast output shape {0} to 1xWxCxH", outputShape));
        return false;
    }

    const auto superdense = isSuperdenseRequired(inputOrder, outputShape, outputOrder, outElemType);
    if (superdense && op.output().getType().isa<VPU::SparseTensorType>()) {
        logCb(formatv("Super-dense mode cannot have sparse output."));
        return false;
    }

    if (op.input().getType().cast<vpux::NDTypeInterface>().getElementType().isF32()) {
        logCb(formatv("F32 input is not supported."));
        return false;
    }

    return true;
}

//
// isSuperdense
//

bool vpux::VPU::NCEPermuteQuantizeOp::isSuperdense() {
    const auto inputOrder = DimsOrder::fromValue(input());
    const auto outTypeND = output().getType().cast<vpux::NDTypeInterface>();
    return isSuperdenseRequired(inputOrder, outTypeND.getShape(), outTypeND.getDimsOrder(), outTypeND.getElementType());
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(NCEPermuteQuantizeOp op) {
    const auto arch = getArch(op);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        return errorAt(op, "Target architecture {0} does not support ODU permutations", arch);
    }

    const auto outElemType = op.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto inputShape = getShape(op.input());
    if (inputShape[Dims4D::Act::C] % alignment != 0) {
        return errorAt(op, "Operation input is expected to be aligned by '{0}' channels, got: '{1}'", alignment,
                       inputShape[Dims4D::Act::C]);
    }

    const auto outputShape = getShape(op.output());
    if (outputShape[Dims4D::Act::C] % alignment != 0) {
        return errorAt(op, "Operation output is expected to be aligned by '{0}' channels, got: '{1}'", alignment,
                       inputShape[Dims4D::Act::C]);
    }

    // Original IE::PermuteQuantizeOp had NCHW input with NWHC output.
    // Derived NCEPermuteQuantizeOp must have NHWC input with NWCH output.
    const auto inputOrder = DimsOrder::fromValue(op.input());
    const auto expectedInOrder = DimsOrder::NHWC;
    if (inputOrder != expectedInOrder) {
        return errorAt(op, "Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inputOrder);
    }

    const auto outputOrder = DimsOrder::fromValue(op.output());
    const auto expectedOutOrder = DimsOrder::NWCH;
    if (outputOrder != expectedOutOrder) {
        return errorAt(op, "Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outputOrder);
    }

    if (op.isSuperdense() && op.output().getType().isa<VPU::SparseTensorType>()) {
        return errorAt(op, "Super-dense mode cannot have sparse output.");
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEPermuteQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEPermuteQuantizeOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inputType = op.input().getType();
    const auto shape = getShape(op.input());
    const auto targetShape = expandShape(shape, op.pad());
    const auto inputOrder = DimsOrder::fromValue(op.input());
    const auto order = DimsOrder::fromAffineMap(op.dstOrder());
    const auto elemType = op.dstElemType();
    const auto superdense = isSuperdenseRequired(inputOrder, ShapeRef(targetShape), order, elemType);
    auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>();
    if (sparseInputType != nullptr && superdense) {
        // Even when input is sparse, output must remain dense because of super-dense mode in ODU.
        inputType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
    }
    const auto typeComponents =
            TypeComponents().setShape(ShapeRef(targetShape)).setDimsOrder(order).setElementType(elemType);
    const auto returnType = inputType.cast<vpux::NDTypeInterface>().changeTypeComponents(typeComponents);

    inferredReturnTypes.push_back(returnType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEPermuteQuantizeOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.setInput(0, DimsOrder::NHWC);
    info.setOutput(0, DimsOrder::NWCH);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEPermuteQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                     vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());
    const auto kernelSize = getIntArrayAttr(getContext(), getKernelSize());
    const auto strides = getIntArrayAttr(getContext(), getStrides());

    // backInferPoolTile satisfies NCEPermuteQuantizeOp demands, let's use it instead of some dedicated tiling.
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, kernelSize, strides, origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    return inputTiling;
}

void vpux::VPU::NCEPermuteQuantizeOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);
}

OutputTiling vpux::VPU::NCEPermuteQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEPermuteQuantizeOp::getKernelSize() {
    return {1, 1};
}

SmallVector<int64_t> vpux::VPU::NCEPermuteQuantizeOp::getStrides() {
    return {1, 1};
}

vpux::VPU::PaddingAttr vpux::VPU::NCEPermuteQuantizeOp::getPad() {
    return padAttr();
}

bool vpux::VPU::NCEPermuteQuantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
}

mlir::LogicalResult vpux::VPU::NCEPermuteQuantizeOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEPermuteQuantizeOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCEPermuteQuantizeOp shouldn't have a serializer");
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEPermuteQuantizeOp::sparsitySupport() {
    using vpux::VPU::SparsitySupport;
    // Super-dense mode does not support ODU sparsity
    if (isSuperdense()) {
        return SparsitySupport::NONE;
    }
    return SparsitySupport::SPARSE_OUTPUTS;
}
