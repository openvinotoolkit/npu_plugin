//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {
bool hasOnlyChannelPadding(ArrayRef<int64_t> pads) {
    // There's a way to calculate this by searching the non-zero padding.
    // But let's keep it simple.
    return pads[Dims4D::Act::N.ind()] == 0 && pads[Dims4D::Act::C.ind()] >= 0 && pads[Dims4D::Act::H.ind()] == 0 &&
           pads[Dims4D::Act::W.ind()] == 0;
}

bool checkSupportedOutputDataType(const mlir::Type outputType) {
    const auto ndType = outputType.cast<vpux::NDTypeInterface>();
    if (ndType == nullptr) {
        return false;
    }
    const auto elemType = ndType.getElementType();
    return elemType.isa<mlir::quant::UniformQuantizedType>() || elemType.isF16() || elemType.isF32();
}

std::vector<int64_t> expandShape(const ShapeRef shape, const int64_t expandedChannels) {
    const std::vector<int64_t> targetShape = {shape[Dims4D::Act::N], expandedChannels, shape[Dims4D::Act::H],
                                              shape[Dims4D::Act::W]};

    return targetShape;
}
}  // namespace

//
// fitIntoCMX
//

bool vpux::VPU::NCEPermuteOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEPermuteOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEPermuteOp::isSupported(IE::PermuteQuantizeOp op, LogCb logCb, bool checkLayout,
                                          bool checkAlignment) {
    // First of all, this operation makes sense only when ODU permutation is supported.
    const auto arch = getArch(op);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        logCb(formatv("Target architecture {0} does not support ODU permutations", arch));
        return false;
    }

    // Check supported output data type : quant uniform or FP16 or FP32.
    if (!checkSupportedOutputDataType(op.getOutput().getType())) {
        logCb(formatv("Unsupported output data type. Got: {0}", op.getOutput().getType()));
        return false;
    }
    // Check padding. NCEPermuteOp supports only padding over channels.
    const auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
    if (padsBegin.size() != 4) {
        logCb(formatv("Only four-dimensional start paddings are supported. Got: {0}", padsBegin));
        return false;
    }
    if (!hasOnlyChannelPadding(padsBegin)) {
        logCb(formatv("Only start padding of channels is supported. Got: {0}", padsBegin));
        return false;
    }
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
    if (padsEnd.size() != 4) {
        logCb(formatv("Only four-dimensional end paddings are supported. Got: {0}", padsBegin));
        return false;
    }
    if (!hasOnlyChannelPadding(padsEnd)) {
        logCb(formatv("Only end padding of channels is supported. Got: {0}", padsEnd));
        return false;
    }

    const auto inputOrder = DimsOrder::fromValue(op.getInput());
    const auto outputOrder = DimsOrder::fromValue(op.getOutput());
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

    const auto outElemType = op.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());

    if (checkAlignment) {
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
    if (superdense && op.getOutput().getType().isa<VPU::SparseTensorType>()) {
        logCb(formatv("Super-dense mode cannot have sparse output."));
        return false;
    }

    if (!op.getInput().getType().cast<vpux::NDTypeInterface>().getElementType().isF16()) {
        logCb(formatv("Only F16 input is supported."));
        return false;
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEPermuteOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        return errorAt(op, "Target architecture {0} does not support ODU permutations", arch);
    }

    const auto outElemType = getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto inputShape = getShape(getInput());
    if (inputShape[Dims4D::Act::W] % alignment != 0) {
        return errorAt(op, "Operation input is expected to be aligned by '{0}' width, got: '{1}'", alignment,
                       inputShape[Dims4D::Act::W]);
    }

    const auto outputShape = getShape(getOutput());
    if (outputShape[Dims4D::Act::W] % alignment != 0) {
        return errorAt(op, "Operation output is expected to be aligned by '{0}' width, got: '{1}'", alignment,
                       outputShape[Dims4D::Act::W]);
    }

    // Original IE::PermuteQuantizeOp had NCHW input with NWHC output.
    // Derived NCEPermuteOp must have NCHW input with NHWC output.
    const auto inputOrder = DimsOrder::fromValue(getInput());
    const auto expectedInOrder = DimsOrder::NCHW;
    if (inputOrder != expectedInOrder) {
        return errorAt(op, "Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inputOrder);
    }

    const auto outputOrder = DimsOrder::fromValue(getOutput());
    const auto expectedOutOrder = DimsOrder::NHWC;
    if (outputOrder != expectedOutOrder) {
        return errorAt(op, "Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outputOrder);
    }

    const auto superdense = VPU::NCESparsity::isSuperdenseRequired(arch, outputOrder, outputShape, outElemType);
    if (superdense && getOutput().getType().isa<VPU::SparseTensorType>()) {
        return errorAt(op, "Super-dense mode cannot have sparse output.");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEPermuteOp::verifyChannels() {
    const auto op = getOperation();

    // We check width here because in following passes a Reorder layer will be added that will generate NWCH order
    const auto outType = getResult().getType().cast<NDTypeInterface>();
    const auto inType = getOperand().getType().cast<NDTypeInterface>();
    if (outType.getRank() != 4 || inType.getRank() != 4) {
        return errorAt(op, "Output activation has unsupported rank: input '{0}' , output '{1}' ", inType.getRank(),
                       outType.getRank());
    }
    const auto outAlignment = getOutputChannelAlignment();
    const auto OW = outType.getShape()[Dims4D::Act::W];
    if (OW % outAlignment != 0) {
        return errorAt(op, "Output width '{0}' is not aligned to '{1}'", OW, outAlignment);
    }
    const auto inAlignment = getInputChannelAlignment();
    const auto IW = inType.getShape()[Dims4D::Act::W];
    if (IW % inAlignment != 0) {
        return errorAt(op, "Input width '{0}' is not aligned to '{1}'", IW, inAlignment);
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEPermuteOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties, mlir::RegionRange regions,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEPermuteOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inputType = op.getInput().getType();
    const auto shape = getShape(op.getInput());
    const auto targetShape = expandShape(shape, op.getExpandedChannels());
    const auto order = DimsOrder::fromAffineMap(op.getDstOrder());
    const auto elemType = op.getDstElemType();
    VPUX_THROW_WHEN(regions.empty(), "NCEPermuteOp::inferReturnTypes got empty list of regions");
    const auto moduleOp = regions[0]->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(moduleOp == nullptr, "NCEPermuteOp::inferReturnTypes region parent is not a ModuleOp");
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

vpux::InputTiling vpux::VPU::NCEPermuteOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origPadding = toPadInfo(VPU::getPaddingAttr(getContext(), 0, 0, 0, 0));
    const auto kernelSize = getIntArrayAttr(getContext(), getKernelSizeVal());
    const auto strides = getIntArrayAttr(getContext(), getStridesVal());

    // backInferPoolTile satisfies NCEPermuteOp demands, let's use it instead of some dedicated tiling.
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, kernelSize, strides, origPadding);
    if (outputTile.axis[Dims4D::Act::C] == 1) {
        inputTiling.tiles[0].shape[Dims4D::Act::C] = origInputShape[Dims4D::Act::C];
    } else {
        // The only case where IC != OC is when expansion is needed for OC
        // and when we tile the operation, expansion only occurs for the last tile
        // which leads to the last outputTile being larger than the last inputTile.
        const auto channelRemainder = origInputShape[Dims4D::Act::C] - outputTile.offsets[Dims4D::Act::C];
        if (channelRemainder < outputTile.shape[Dims4D::Act::C] && channelRemainder > 0) {
            inputTiling.tiles[0].shape[Dims4D::Act::C] = channelRemainder;
        } else {
            inputTiling.tiles[0].shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        }
    }
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    return inputTiling;
}

void vpux::VPU::NCEPermuteOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& outputTile) {
    setExpandedChannelsAttr(getIntAttr(getContext(), outputTile.shape[Dims4D::Act::C]));
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEPermuteOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEPermuteOp::getKernelSizeVal() {
    return {1, 1};
}

SmallVector<int64_t> vpux::VPU::NCEPermuteOp::getStridesVal() {
    return {1, 1};
}

vpux::VPU::PaddingAttr vpux::VPU::NCEPermuteOp::getPad() {
    return VPU::getPaddingAttr(getContext(), 0, 0, 0, 0);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEPermuteOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEPermuteOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, vpux::VPU::PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    return VPU::getNCEExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                                    distributionMode, numTiles, numClusters, alignment, kernel, pad,
                                                    stride, uniformDistributedSegments);
}

bool VPU::NCEPermuteOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    return VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile);
}

bool VPU::NCEPermuteOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEPermuteOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEPermuteOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEPermuteOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape()[Dims4D::Act::C], strategy);
    return fitIntoCMX(getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy),
                      getDistributedOutputTypeFromOp(nceOp, nceOp.getOutput().getType(), numClusters, strategy),
                      reservedMem);
}

mlir::LogicalResult vpux::VPU::NCEPermuteOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEPermuteOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto arch = getArch(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        VPUX_THROW("NCEPermuteOp is not supported for {0}", arch);
    case VPU::ArchKind::VPUX37XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
