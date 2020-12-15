//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AddOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AddOpAdaptor add(operands, attrs);
    if (mlir::failed(add.verify(loc))) {
        return ::mlir::failure();
    }

    if (add.auto_broadcast().getValue() == vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        auto outType = add.input1().getType().cast<mlir::RankedTensorType>();
        inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType());
    } else {
        // Broadcasting input shapes
        auto inType1 = add.input1().getType().cast<mlir::RankedTensorType>();
        auto inType2 = add.input2().getType().cast<mlir::RankedTensorType>();
        auto inShape1 = inType1.getShape();
        auto inShape2 = inType2.getShape();
        size_t maxShape = std::max(inShape1.size(), inShape2.size());

        std::vector<int64_t> broadcastingShape;
        broadcastingShape.reserve(maxShape);
        for (auto it1 = inShape1.rbegin(), it2 = inShape2.rbegin(); it1 != inShape1.rend() || it2 != inShape2.rend();) {
            if (it1 == inShape1.rend() && it2 == inShape2.rend())
                break;

            if (it1 != inShape1.rend() && it2 != inShape2.rend()) {
                broadcastingShape.push_back(std::max(*it1, *it2));
                ++it1;
                ++it2;
            } else if (it1 != inShape1.rend() && it2 == inShape2.rend()) {
                broadcastingShape.push_back(*it1);
                ++it1;
            }
            if (it1 == inShape1.rend() && it2 != inShape2.rend()) {
                broadcastingShape.push_back(*it2);
                ++it2;
            }
        }
        std::reverse(std::begin(broadcastingShape), std::end(broadcastingShape));

        inferredReturnShapes.emplace_back(makeArrayRef(broadcastingShape), inType1.getElementType());
    }
    return mlir::success();
}
