//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConstantFoldingPass
//

class ConstantFoldingPass final : public Const::ConstantFoldingBase<ConstantFoldingPass> {
public:
    explicit ConstantFoldingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConstantFoldingPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](Const::DeclareOp origOp) {
        _log.trace("Folding constant at location '{0}'", origOp.getLoc());

        mlir::OpBuilder builder(origOp);

        const auto content = origOp.content();
        const auto contentType = content.getType();
        const auto contentElemType = contentType.getElementType();

        const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
        std::vector<char> tempBuf(bufSize);
        content.copyTo(makeMutableArrayRef(tempBuf.data(), bufSize));

        auto rankedTensorType = contentType.cast<mlir::RankedTensorType>();

        if (auto qtype = contentElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            rankedTensorType =
                    contentType.changeElemType(normalizeQuantStorageType(qtype)).cast<mlir::RankedTensorType>();
        }

        const auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(rankedTensorType, tempBuf);

        const auto newOp =
                builder.create<Const::DeclareOp>(origOp.getLoc(), origOp.getType(), Const::ContentAttr::get(denseAttr));
        origOp.replaceAllUsesWith(newOp);

        origOp.erase();
    });
}

}  // namespace

//
// createConstantFoldingPass
//

std::unique_ptr<mlir::Pass> vpux::Const::createConstantFoldingPass(Logger log) {
    return std::make_unique<ConstantFoldingPass>(log);
}
