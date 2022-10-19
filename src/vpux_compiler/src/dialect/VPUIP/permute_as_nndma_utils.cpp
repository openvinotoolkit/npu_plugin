//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/permute_as_nndma_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"

using namespace vpux;

SmallVector<Shape> vpux::VPUIP::getDMASubShape(ShapeRef shape) {
    const auto N = shape[Dims4D::Act::N];
    const auto C = shape[Dims4D::Act::C];

    auto numberDMAs = divUp(C, DMA_MAX_NUMBER_PLANES);
    auto subShape = Shape(SmallVector<int64_t>{C, N});
    if (numberDMAs > 1) {
        subShape[Dims4D::Act::N] = DMA_MAX_NUMBER_PLANES;
        SmallVector<Shape> outputShapes(numberDMAs - 1, subShape);
        subShape[Dims4D::Act::N] = C - DMA_MAX_NUMBER_PLANES * (numberDMAs - 1);
        outputShapes.push_back(subShape);
        return outputShapes;
    }

    return SmallVector<Shape>{subShape};
}

Optional<Shape> vpux::VPUIP::getPermuteDMAOutputShape(NDTypeInterface inType, NDTypeInterface outType,
                                                      DimsOrder memPerm, vpux::Logger log) {
    auto inShape = inType.getShape();
    auto inOrder = inType.getDimsOrder();
    auto outShape = outType.getShape();

    Shape inputRealShape(inShape.size());
    for (size_t idx = 0; idx < inShape.size(); idx++) {
        inputRealShape[Dim(idx)] = inShape[inOrder.dimAt(idx)];
    }

    Shape newOutShape;
    int64_t shapeSize = 1;
    for (size_t idx = 0; idx < inputRealShape.size(); idx++) {
        if (shapeSize != 1 && inputRealShape[memPerm.dimAt(idx)] > DMA_MAX_NUMBER_PLANES) {
            log.trace("Can't convert Permute with number plane > {0}, inshape {1}, outshape {2}, memPerm {3}.",
                      DMA_MAX_NUMBER_PLANES, inShape, outShape, memPerm);
            return None;
        }
        shapeSize *= inputRealShape[memPerm.dimAt(idx)];

        if (idx + 1 == inputRealShape.size() || memPerm.dimAt(idx).ind() + 1 != memPerm.dimAt(idx + 1).ind()) {
            if (shapeSize != 1) {
                newOutShape.push_back(shapeSize);
            }
            shapeSize = 1;
        }
    }

    if (newOutShape.size() != 2) {
        log.trace("Can't convert Permute to DMA with inshape {0}, outshape {1}, memPerm {2}.", inShape, outShape,
                  memPerm);
        return None;
    }

    return newOutShape;
}

Optional<SmallVector<Shape>> vpux::VPUIP::getPermuteDMASubShapes(VPUIP::PermuteDMAOp permuteOp, vpux::Logger log) {
    auto inType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outType = permuteOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!permuteOp.mem_perm().hasValue() || !permuteOp.mem_perm().getValue().isPermutation()) {
        log.trace("PermuteOp {0} doesn't support permutation.", permuteOp->getLoc());
        return None;
    }
    const auto memPerm = DimsOrder::fromAffineMap(permuteOp.mem_perm().getValue());

    auto newOutShape = getPermuteDMAOutputShape(inType, outType, memPerm, log);
    if (!newOutShape.hasValue()) {
        return None;
    }

    return getDMASubShape(newOutShape.getValue());
}

Optional<SmallVector<Shape>> vpux::VPUIP::getPermuteDMASubShapes(VPUIP::PermuteUPAOp permuteUPAOp, vpux::Logger log) {
    auto inType = permuteUPAOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outType = permuteUPAOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!permuteUPAOp.order_value().isPermutation()) {
        log.trace("PermuteOp {0} doesn't support permutation.", permuteUPAOp->getLoc());
        return None;
    }
    const auto memPerm = DimsOrder::fromAffineMap(permuteUPAOp.order_value());

    auto newOutShape = getPermuteDMAOutputShape(inType, outType, memPerm, log);
    if (!newOutShape.hasValue()) {
        return None;
    }

    return getDMASubShape(newOutShape.getValue());
}

int64_t vpux::VPUIP::getDstStride(SmallVector<Shape> shapes) {
    int64_t dstStride = 0;
    for (auto shape : shapes) {
        dstStride += shape[Dims4D::Act::N];
    }
    return dstStride;
}

/**
 * Cost function to evaluate whether it's beneficial to implement the operation using DMA rather than UPA for
 * operations like MemPermute.
 * @return true if it's beneficial for using DMA, otherwise false.
 */
bool vpux::VPUIP::isBeneficialForUsingDMA(mlir::Operation* op, vpux::Logger log) {
    if (auto permuteUPAOp = mlir::cast<VPUIP::PermuteUPAOp>(op)) {
        auto subShapes = getPermuteDMASubShapes(permuteUPAOp, log);
        if (!subShapes.hasValue()) {
            return false;
        }

        // This is a empirical value to set limitation for DMA number.
        // In some specific case, for example: 1x8x256x256 #NHWC  ->  1x8x256x256 #NCHW
        // This permuteUPA should be replaced with 256 DMA. Each DMA just with size 8x256. It is inefficient.
        // It's supposed to get the related DMA and UPA cost by VPUNN in future, please refer ticket #41221.
        if (subShapes.getValue().size() > PER_PERMUTE_MAX_DMA_NUMBER) {
            return false;
        }

        return llvm::all_of(subShapes.getValue(), [](ShapeRef shape) {
            const auto C = shape[Dims4D::Act::C];
            // This is a empirical value to set limitation for the src_width in the DMA descriptor. For dst_with
            // fixed as element size, with smaller src_width, performance tends to be better. It's supposed to get the
            // related DMA and UPA cost by VPUNN in future, please refer ticket #41221.
            return C < PERMUTE_DMA_MAX_LENTH;
        });
    }
    return false;
}

bool vpux::VPUIP::isCombineAtFront(ShapeRef shape, DimsOrder order) {
    for (size_t idx = 0; idx < shape.size(); idx++) {
        if (shape[order.dimAt(idx)] == 1) {
            continue;
        }
        return shape[order.dimAt(idx)] <= DMA_MAX_NUMBER_PLANES;
    }
    return false;
}
