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

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConvertOpAdaptor cvt(operands, attrs);
    if (mlir::failed(cvt.verify(loc))) {
        return ::mlir::failure();
    }

    auto inType = cvt.input().getType().cast<mlir::RankedTensorType>();
    auto dstElemType = cvt.dstType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);

    return mlir::success();
}

namespace IE_Convert {
namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/convert.hpp.inc>

}  // namespace
}  // namespace IE_Convert

void vpux::IE::ConvertOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                      mlir::MLIRContext* context) {
    IE_Convert::populateWithGenerated(context, patterns);
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(ArrayRef<mlir::Attribute>) {
    if (inputType() == outputType()) {
        return input();
    }

    return nullptr;
}
