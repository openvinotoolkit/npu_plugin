//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

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
    const auto& top = pads.getTop().getInt();
    const auto& bottom = pads.getBottom().getInt();
    const auto& left = pads.getLeft().getInt();
    const auto& right = pads.getRight().getInt();

    const std::vector<int64_t> targetShape = {shape[Dims4D::Act::N], shape[Dims4D::Act::C],
                                              shape[Dims4D::Act::H] + top + bottom,
                                              shape[Dims4D::Act::W] + left + right};

    return targetShape;
}
}  // namespace

//
// fitIntoCMX
//

bool vpux::VPU::NCEPermuteQuantizeOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                                 Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEPermuteQuantizeOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEPermuteQuantizeOp::isSupported(IE::PermuteQuantizeOp op, LogCb logCb, bool checkLayout,
                                                  bool checkChannelAlignment) {
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
    const auto outputOrder = DimsOrder::fromValue(op.output());
    if (checkLayout) {
        const auto expectedInOrder = DimsOrder::NCHW;
        if (inputOrder != expectedInOrder) {
            logCb(formatv("Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inputOrder));
            return false;
        }
        const auto expectedOutOrder = DimsOrder::NHWC;
        if (outputOrder != expectedOutOrder) {
            logCb(formatv("Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outputOrder));
            return false;
        }
    }

    const auto outElemType = op.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto inputShape = getShape(op.input());
    const auto outputShape = getShape(op.output());

    if (checkChannelAlignment) {
        const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
        if (!IE::isShapeCompatibleWithODUPermute(inputShape, alignment)) {
            logCb(formatv("Cannot cast input shape {0} to 1xWxCxH", inputShape));
            return false;
        }

        if (!IE::isShapeCompatibleWithODUPermute(outputShape, alignment)) {
            logCb(formatv("Cannot cast output shape {0} to 1xWxCxH", outputShape));
            return false;
        }
    }

    const auto superdense = VPU::NCESparsity::isSuperdenseRequired(arch, outputOrder, outputShape, outElemType);
    if (superdense && op.output().getType().isa<VPU::SparseTensorType>()) {
        logCb(formatv("Super-dense mode cannot have sparse output."));
        return false;
    }

    if (!op.input().getType().cast<vpux::NDTypeInterface>().getElementType().isF16()) {
        logCb(formatv("Only F16 input is supported."));
        return false;
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEPermuteQuantizeOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        return errorAt(op, "Target architecture {0} does not support ODU permutations", arch);
    }

    const auto outElemType = output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto inputShape = getShape(input());
    if (inputShape[Dims4D::Act::C] % alignment != 0) {
        return errorAt(op, "Operation input is expected to be aligned by '{0}' channels, got: '{1}'", alignment,
                       inputShape[Dims4D::Act::C]);
    }

    const auto outputShape = getShape(output());
    if (outputShape[Dims4D::Act::C] % alignment != 0) {
        return errorAt(op, "Operation output is expected to be aligned by '{0}' channels, got: '{1}'", alignment,
                       outputShape[Dims4D::Act::C]);
    }

    // Original IE::PermuteQuantizeOp had NCHW input with NWHC output.
    // Derived NCEPermuteQuantizeOp must have NHWC input with NWCH output.
    const auto inputOrder = DimsOrder::fromValue(input());
    const auto expectedInOrder = DimsOrder::NHWC;
    if (inputOrder != expectedInOrder) {
        return errorAt(op, "Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inputOrder);
    }

    const auto outputOrder = DimsOrder::fromValue(output());
    const auto expectedOutOrder = DimsOrder::NWCH;
    if (outputOrder != expectedOutOrder) {
        return errorAt(op, "Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outputOrder);
    }

    const auto superdense = VPU::NCESparsity::isSuperdenseRequired(arch, outputOrder, outputShape, outElemType);
    if (superdense && output().getType().isa<VPU::SparseTensorType>()) {
        return errorAt(op, "Super-dense mode cannot have sparse output.");
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEPermuteQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange regions, mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEPermuteQuantizeOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inputType = op.input().getType();
    const auto shape = getShape(op.input());
    const auto targetShape = expandShape(shape, op.pad());
    const auto order = DimsOrder::fromAffineMap(op.dstOrder());
    const auto elemType = op.dstElemType();
    VPUX_THROW_WHEN(regions.empty(), "NCEPermuteQuantizeOp::inferReturnTypes got empty list of regions");
    const auto moduleOp = regions[0]->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(moduleOp == nullptr, "NCEPermuteQuantizeOp::inferReturnTypes region parent is not a ModuleOp");
    const auto arch = VPU::getArch(moduleOp);
    const auto superdense = VPU::NCESparsity::isSuperdenseRequired(arch, order, ShapeRef(targetShape), elemType);
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
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEPermuteQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                     vpux::Logger log) {
    const auto origInputShape = getShape(input());
    const auto origPadding = toPadInfo(pad());
    const auto kernelSize = getIntArrayAttr(getContext(), getKernelSizeVal());
    const auto strides = getIntArrayAttr(getContext(), getStridesVal());

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

mlir::FailureOr<OutputTiling> vpux::VPU::NCEPermuteQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEPermuteQuantizeOp::getKernelSizeVal() {
    return {1, 1};
}

SmallVector<int64_t> vpux::VPU::NCEPermuteQuantizeOp::getStridesVal() {
    return {1, 1};
}

bool vpux::VPU::NCEPermuteQuantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEPermuteQuantizeOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, vpux::VPU::PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    const auto actTensorDistrModeAttr = DistributionModeAttr::get(getContext(), distributionMode);
    DistributedTensorAttr distributedActivationTensorAttr = DistributedTensorAttr::get(
            getContext(), actTensorDistrModeAttr, numTiles, kernel, pad, stride, numClusters, alignment,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto perClusterMemoryShapes = vpux::getIntArrayOfArray(
            getContext(), VPU::getPerClusterMemoryShapes(shape, distributedActivationTensorAttr));
    auto perClusterMemoryOffsets = vpux::getIntArrayOfArray(
            getContext(), VPU::getPerClusterMemoryShapeOffsets(shape, distributedActivationTensorAttr));

    // Unlike other NCE ops, PermuteQuantize needs to compute the overlap section in both neighbouring clusters
    // when it has overlapped distribution mode for the op to produce the correct output.
    if (distributionMode == DistributionMode::OVERLAPPED) {
        return vpux::VPU::DistributedTensorAttr::get(getContext(), actTensorDistrModeAttr, numTiles, nullptr, nullptr,
                                                     nullptr, numClusters, alignment, uniformDistributedSegments,
                                                     perClusterMemoryShapes, perClusterMemoryOffsets,
                                                     perClusterMemoryShapes, perClusterMemoryOffsets, nullptr);
    }

    auto perClusterComputeShapes = vpux::getIntArrayOfArray(
            getContext(), VPU::getPerClusterComputeShapes(shape, distributedActivationTensorAttr));
    auto perClusterComputeOffsets = vpux::getIntArrayOfArray(
            getContext(), VPU::getPerClusterComputeShapeOffsets(shape, distributedActivationTensorAttr));

    return vpux::VPU::DistributedTensorAttr::get(getContext(), actTensorDistrModeAttr, numTiles, nullptr, nullptr,
                                                 nullptr, numClusters, alignment, uniformDistributedSegments,
                                                 perClusterComputeShapes, perClusterComputeOffsets,
                                                 perClusterMemoryShapes, perClusterMemoryOffsets, nullptr);
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
        VPUX_THROW("NCEPermuteQuantizeOp is not supported for {0}", arch);
    case VPU::ArchKind::VPUX37XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
