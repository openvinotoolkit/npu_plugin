//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

//
// RemoveDuplicatesPass
//

class RemoveDuplicatesPass final : public IE::RemoveDuplicatesBase<RemoveDuplicatesPass> {
public:
    explicit RemoveDuplicatesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveDuplicatesPass::safeRunOnFunc() {
    auto func = getFunction();

    using OpDef = std::tuple<mlir::OperationName, mlir::Attribute, mlir::Value, mlir::Type>;
    llvm::DenseMap<OpDef, SmallVector<mlir::Operation*>> ops;

    func.walk([&](IE::LayerOpInterface layer) {
        if (layer->getNumOperands() != 1) {
            return;
        }
        if (layer->getNumResults() != 1) {
            return;
        }

        const auto def = OpDef(layer->getName(), layer->getAttrDictionary(), layer->getOperand(0),
                               layer->getResult(0).getType());
        ops[def].push_back(layer);
    });

    for (const auto& p : ops) {
        const auto& duplicates = makeArrayRef(p.second);

        if (duplicates.size() == 1) {
            continue;
        }

        auto* finalOp = duplicates.front();

        for (auto* duplOp : duplicates.drop_front(1)) {
            duplOp->replaceAllUsesWith(finalOp->getResults());
            duplOp->erase();
        }
    }
}

}  // namespace

//
// createRemoveDuplicatesPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveDuplicatesPass(Logger log) {
    return std::make_unique<RemoveDuplicatesPass>(log);
}
