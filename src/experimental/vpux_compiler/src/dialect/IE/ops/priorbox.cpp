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
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <ngraph/op/prior_box.hpp>

using namespace vpux;

namespace {

ngraph::op::PriorBoxAttrs getNGraphPriorBoxAttrs(IE::PriorBoxOpAdaptor priorBox) {
    const auto getNGraphFPArray = [](mlir::ArrayAttr attr) {
        return to_std_vector(parseFPArrayAttr(attr) | transformed([](double val) {
                                 return checked_cast<float>(val);
                             }));
    };

    ngraph::op::PriorBoxAttrs priorBoxAttrs;
    priorBoxAttrs.min_size = getNGraphFPArray(priorBox.min_size());
    priorBoxAttrs.max_size = getNGraphFPArray(priorBox.max_size());
    priorBoxAttrs.aspect_ratio = getNGraphFPArray(priorBox.aspect_ratio());
    priorBoxAttrs.density = getNGraphFPArray(priorBox.density());
    priorBoxAttrs.fixed_ratio = getNGraphFPArray(priorBox.fixed_ratio());
    priorBoxAttrs.fixed_size = getNGraphFPArray(priorBox.fixed_size());
    priorBoxAttrs.variance = getNGraphFPArray(priorBox.variance());
    priorBoxAttrs.clip = priorBox.clip().getValue();
    priorBoxAttrs.flip = priorBox.flip().getValue();
    priorBoxAttrs.step = priorBox.step().getValue().convertToFloat();
    priorBoxAttrs.offset = priorBox.offset().getValue().convertToFloat();
    priorBoxAttrs.scale_all_sizes = priorBox.scale_all_sizes().getValue();

    return priorBoxAttrs;
}

}  // namespace

mlir::LogicalResult vpux::IE::PriorBoxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PriorBoxOpAdaptor priorBox(operands, attrs);
    if (mlir::failed(priorBox.verify(loc))) {
        return mlir::failure();
    }

    auto outputSize = priorBox.output_size().getDefiningOp<mlir::ConstantOp>();
    if (outputSize == nullptr) {
        return mlir::failure();
    }

    const auto denseElementArray = outputSize.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    const auto elementsRange = denseElementArray.getValues<int64_t>();
    VPUX_THROW_UNLESS(denseElementArray.size() == 2, "output_size of priorbox should be 2");

    const auto priorBoxAttrs = getNGraphPriorBoxAttrs(priorBox);
    const auto numPriors = ngraph::op::PriorBox::number_of_priors(priorBoxAttrs);

    SmallVector<int64_t> outShape{2, 4 * numPriors};
    outShape[1] *= *elementsRange.begin();
    outShape[1] *= *(elementsRange.begin() + 1);

    inferredReturnShapes.emplace_back(outShape, mlir::Float32Type::get(ctx));
    return mlir::success();
}
