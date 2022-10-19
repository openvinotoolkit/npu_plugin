//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool strictInfer) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dst_order);
    const auto inType = input.getType().cast<mlir::RankedTensorType>();

    const auto inShape = getShape(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, mem_perm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outDesc = IE::getTensorAttr(dst_order, strictInfer ? IE::getMemorySpace(inType) : nullptr);

    inferredReturnShapes.emplace_back(outShape.raw(), inType.getElementType(), outDesc);
}
