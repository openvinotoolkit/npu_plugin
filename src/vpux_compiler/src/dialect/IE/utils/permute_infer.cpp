//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool useInMemSpace) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromPermutationAffineMap(dst_order);
    const auto inType = input.getType().cast<mlir::RankedTensorType>();

    const auto inShape = getShape(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, mem_perm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outDesc = IE::getTensorAttr(dst_order, useInMemSpace ? IE::getMemorySpace(inType) : nullptr);

    inferredReturnShapes.emplace_back(outShape.raw(), inType.getElementType(), outDesc);
}
