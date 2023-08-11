//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output) {
    if (this->is_inplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, Byte(0));
    }

    return fitIntoCMX(input1, input2, output, Byte(0));
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output, Byte reservedMem) {
    if (this->is_inplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, reservedMem);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize(), output.getTotalAllocSize()};

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEEltwiseOp::isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                          LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    if (op->getNumOperands() != 2) {
        return false;
    }
    auto input1Type = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto input2Type = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    return vpux::VPU::isNCEEltwiseSupported(getArch(op), input1Type, input2Type, outputType, allowDifferentScales,
                                            allowDifferentZp, checkLayout, checkChannelAlignment, logCb);
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEEltwiseOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto shape1 = getShape(op.input1());
    const auto shape2 = getShape(op.input2());

    if (shape1 != shape2) {
        return errorAt(loc, "Broadcasting is not supported for {0} operation", NCEEltwiseOp::getOperationName());
    }

    auto inputType = op.input1().getType();
    if (auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        inputType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
    }

    inferredReturnTypes.push_back(inputType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEEltwiseOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEEltwiseOp::getKernelSize() {
    return {1, 1};
}

SmallVector<int64_t> vpux::VPU::NCEEltwiseOp::getStrides() {
    return {1, 1};
}

vpux::VPU::PaddingAttr vpux::VPU::NCEEltwiseOp::getPad() {
    return VPU::getPaddingAttr(getContext(), PadInfo(0, 0, 0, 0));
}

bool vpux::VPU::NCEEltwiseOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    if (this->is_inplace().value_or(false)) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NCEEltwiseOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("NCEEltwiseOp shouldn't have a serializer");
}

bool vpux::VPU::NCEEltwiseOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

bool vpux::VPU::NCEEltwiseOp::availableSingleMerge() {
    // if both operand of eltwise are different and cannot be merged with blocks
    // come before its operands, it would be invalid VF subgraph, reject merging
    return input1() == input2();
};

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEEltwiseOp::sparsitySupport() {
    const auto arch = getArch(getOperation());
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return VPU::SparsitySupport::NONE;
    case VPU::ArchKind::VPUX37XX:
        // TODO E#66913: enable input sparsity support once inputs are aligned with respect to storage element size
        return VPU::SparsitySupport::SPARSE_OUTPUTS;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEEltwiseOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::NCEEltwiseOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // Do nothing
}

OutputTiling vpux::VPU::NCEEltwiseOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
