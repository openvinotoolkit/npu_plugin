//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

int64_t extractMaxOutputBoxesPerClass(IE::NonMaxSuppressionOpAdaptor nms) {
    int64_t maxOutputBoxesPerClass = 0;  // default value

    if (nms.max_output_boxes_per_class() != nullptr) {
        auto maxBoxesConst = nms.max_output_boxes_per_class().getDefiningOp<Const::DeclareOp>();
        if (maxBoxesConst != nullptr) {
            const auto maxBoxesContent = maxBoxesConst.content();
            if (maxBoxesContent.isSplat()) {
                return maxBoxesContent.getSplatValue<int64_t>();
            }
        }
    }
    if (nms.max_output_boxes_per_class_valueAttr() != nullptr) {
        return nms.max_output_boxes_per_class_valueAttr().getValue().getSExtValue();
    }

    return maxOutputBoxesPerClass;
}

double extractNMSAttrValue(mlir::Value constName, mlir::FloatAttr attrName) {
    double attrValue = 0.0f;
    if (constName != nullptr) {
        vpux::Const::DeclareOp attrConst = constName.getDefiningOp<Const::DeclareOp>();
        if (attrConst != nullptr) {
            vpux::Const::Content attrContent = attrConst.content();
            if (attrContent.isSplat()) {
                attrValue = attrContent.getSplatValue<float>();
            }
        }
    } else if (attrName != nullptr) {
        attrValue = attrName.getValueAsDouble();
    }
    return attrValue;
}

}  // namespace

mlir::LogicalResult vpux::IE::NonMaxSuppressionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonMaxSuppressionOpAdaptor nms(operands, attrs);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = nms.in_box_scores().getType().cast<mlir::ShapedType>();
    mlir::Type outType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);

    const auto maxOutputBoxesPerClass = extractMaxOutputBoxesPerClass(nms);

    const auto numBatches = inType.getShape()[0];
    const auto numClasses = inType.getShape()[1];
    const auto numBoxes = inType.getShape()[2];
    const auto minBoxes = std::min(numBoxes, maxOutputBoxesPerClass);
    const SmallVector<int64_t> outShape{minBoxes * numBatches * numClasses, 3};
    const SmallVector<int64_t> validOutputsShape{1};
    inferredReturnShapes.emplace_back(outShape, outType);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(validOutputsShape, outType);

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    using mlir::OpRewritePattern<IE::NonMaxSuppressionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp nmsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::NonMaxSuppressionOp nmsOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (nmsOp.max_output_boxes_per_class_value().hasValue() && nmsOp.iou_threshold_value().hasValue() &&
        nmsOp.score_threshold_value().hasValue() && nmsOp.soft_nms_sigma_value().hasValue()) {
        return mlir::failure();
    }

    int64_t maxBoxesPerClassValue = extractMaxOutputBoxesPerClass(nmsOp);

    double iouThresholdValue = extractNMSAttrValue(nmsOp.iou_threshold(), nmsOp.iou_threshold_valueAttr());

    double scoreThresholdValue = extractNMSAttrValue(nmsOp.score_threshold(), nmsOp.score_threshold_valueAttr());

    double softNMSSigmaValue = extractNMSAttrValue(nmsOp.soft_nms_sigma(), nmsOp.soft_nms_sigma_valueAttr());

    rewriter.replaceOpWithNewOp<IE::NonMaxSuppressionOp>(
            nmsOp, nmsOp.in_box_coords(), nmsOp.in_box_scores(), nullptr, nullptr, nullptr, nullptr,
            nmsOp.box_encoding(), nmsOp.sort_result_descending(), rewriter.getI64IntegerAttr(maxBoxesPerClassValue),
            rewriter.getF64FloatAttr(iouThresholdValue), rewriter.getF64FloatAttr(scoreThresholdValue),
            rewriter.getF64FloatAttr(softNMSSigmaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::NonMaxSuppressionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
