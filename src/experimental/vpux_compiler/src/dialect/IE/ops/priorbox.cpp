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

#include "ngraph/op/prior_box.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

static ngraph::op::PriorBoxAttrs convertToNgraphPriorBoxAttr(IE::PriorBoxOpAdaptor priorBox) {
    std::function<std::vector<float>(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToFloatVector =
            [](mlir::ArrayAttr&& arrayAttr) {
                std::vector<float> result;
                for (auto&& a : arrayAttr)
                    result.push_back(a.dyn_cast<mlir::FloatAttr>().getValueAsDouble());
                return result;
            };

    ngraph::op::PriorBoxAttrs priorBoxAttrs;

    priorBoxAttrs.min_size = convertArrayAttrToFloatVector(priorBox.min_size());
    priorBoxAttrs.max_size = convertArrayAttrToFloatVector(priorBox.max_size());
    priorBoxAttrs.aspect_ratio = convertArrayAttrToFloatVector(priorBox.aspect_ratio());
    priorBoxAttrs.density = convertArrayAttrToFloatVector(priorBox.density());
    priorBoxAttrs.fixed_ratio = convertArrayAttrToFloatVector(priorBox.fixed_ratio());
    priorBoxAttrs.fixed_size = convertArrayAttrToFloatVector(priorBox.fixed_size());
    priorBoxAttrs.variance = convertArrayAttrToFloatVector(priorBox.variance());
    priorBoxAttrs.clip = priorBox.clip().getValue();
    priorBoxAttrs.flip = priorBox.flip().getValue();
    priorBoxAttrs.step = priorBox.step().getValue().convertToFloat();
    priorBoxAttrs.offset = priorBox.offset().getValue().convertToFloat();
    priorBoxAttrs.scale_all_sizes = priorBox.scale_all_sizes().getValue();
    return priorBoxAttrs;
}

mlir::LogicalResult vpux::IE::PriorBoxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PriorBoxOpAdaptor priorBox(operands, attrs);
    if (mlir::failed(priorBox.verify(loc))) {
        return ::mlir::failure();
    }

    auto priorBoxAttrs = convertToNgraphPriorBoxAttr(priorBox);
    auto numPriors = ngraph::op::PriorBox::number_of_priors(priorBoxAttrs);

    mlir::SmallVector<int64_t, 2> outShape{2, 4 * numPriors};

    auto outputSize = priorBox.output_size().getDefiningOp<mlir::ConstantOp>();
    if (outputSize) {
        auto denseElementArray = outputSize.value().dyn_cast<mlir::DenseElementsAttr>();
        if (denseElementArray) {
            auto elementsRange = denseElementArray.getValues<int64_t>();
            VPUX_THROW_UNLESS(elementsRange.end() - elementsRange.begin() == 2, "output_size of priorbox should be 2");
            outShape[1] *= *elementsRange.begin();
            outShape[1] *= *(elementsRange.begin() + 1);
            inferredReturnShapes.emplace_back(outShape, mlir::Float32Type::get(ctx));
            return mlir::success();
        }
    }
    return ::mlir::failure();
}
