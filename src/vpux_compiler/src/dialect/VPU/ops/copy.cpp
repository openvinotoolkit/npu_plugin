//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::CopyOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::CopyOpAdaptor copyOp(operands, attrs);
    if (mlir::failed(copyOp.verify(loc))) {
        return mlir::failure();
    }

    const auto ndInType = copyOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
    if (ndInType == nullptr) {
        return errorAt(loc, "IE::CopyOp operand must have vpux::NDTypeInterface type");
    }

    const auto outMemSpace = copyOp.out_mem_space();
    const auto outType = ndInType.changeMemSpace(outMemSpace);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::CopyOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseCopies final : public mlir::OpRewritePattern<VPU::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseCopies::matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerCopyOp = origOp.input().getDefiningOp<VPU::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, producerCopyOp.input(), origOp.out_mem_spaceAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::CopyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseCopies>(ctx);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::CopyOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::CopyParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CopyParams});
}
