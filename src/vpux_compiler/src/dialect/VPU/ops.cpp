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

#include "vpux/compiler/utils/asm.hpp"

using namespace vpux;

mlir::LogicalResult VPU::isDistributedCastCompatible(VPU::DistributedTensorType inDistributedType,
                                                     VPU::DistributedTensorType outDistributedType) {
    const auto loc = mlir::UnknownLoc::get(inDistributedType.getContext());

    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        return errorAt(loc, "Mismatch between tensor shapes for input ({0}) and output ({1}).",
                       inDistributedType.getShape(), outDistributedType.getShape());
    }

    if (inDistributedType.getElementType() != outDistributedType.getElementType()) {
        return errorAt(loc, "Mismatch between tensor element types for input ({0}) and output ({1}).",
                       inDistributedType.getElementType(), outDistributedType.getElementType());
    }

    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        return errorAt(loc, "Mismatch between tensor order for input ({0}) and output ({1}).",
                       inDistributedType.getOrder(), outDistributedType.getOrder());
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        return errorAt(loc, "Mismatch between tensor memspaces for input ({0}) and output ({1}).",
                       inDistributedType.getMemSpace(), outDistributedType.getMemSpace());
    }

    const auto inDistributionAttr = inDistributedType.getDistribution();
    const auto outDistributionAttr = outDistributedType.getDistribution();

    if (inDistributionAttr.num_clusters() != outDistributionAttr.num_clusters()) {
        return errorAt(loc, "Mismatch between tensor number of clusters for input ({0}) and output ({1}).",
                       inDistributionAttr.num_clusters(), outDistributionAttr.num_clusters());
    }

    const auto inDistributionMode = inDistributionAttr.mode().getValue();
    const auto outDistributionMode = outDistributionAttr.mode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::areDistributionModesCompatible(inDistributionMode, outDistributionMode).failed()) {
            return errorAt(loc, "Incompatible distribution modes for input ({0}) and output ({1}).",
                           VPU::stringifyDistributionMode(inDistributionMode),
                           VPU::stringifyDistributionMode(outDistributionMode));
        }
    }

    if (inDistributionAttr.alignment() != outDistributionAttr.alignment()) {
        return errorAt(loc, "Mismatch between alignments for input ({0}) and output ({1}).",
                       inDistributionAttr.alignment(), outDistributionAttr.alignment());
    }

    return mlir::success();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/generated/ops.cpp.inc>
