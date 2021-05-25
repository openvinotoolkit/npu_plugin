//
// Copyright 2020 Intel Corporation.
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

    auto outputSizeConst = priorBox.output_size().getDefiningOp<ConstantInterface>();
    if (outputSizeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for output_size");
    }

    const auto outputSize = outputSizeConst.getContent().getValues<int64_t>();
    if (outputSize.size() != 2) {
        return errorAt(loc, "output_size of priorbox should be 2");
    }

    const auto priorBoxAttrs = getNGraphPriorBoxAttrs(priorBox);
    const auto numPriors = ngraph::op::PriorBox::number_of_priors(priorBoxAttrs);

    SmallVector<int64_t> outShape{2, 4 * numPriors};
    outShape[1] *= outputSize[0];
    outShape[1] *= outputSize[1];

    inferredReturnShapes.emplace_back(outShape, mlir::Float32Type::get(ctx));

    return mlir::success();
}
