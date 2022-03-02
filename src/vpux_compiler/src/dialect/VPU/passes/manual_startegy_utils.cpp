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

#include "vpux/compiler/dialect/VPU/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Block.h>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// ManualStrategyUtilsPass
//

class ManualStrategyUtilsPass final : public ManualStrategyUtilsBase<ManualStrategyUtilsPass> {
public:
    explicit ManualStrategyUtilsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ManualStrategyUtilsPass::safeRunOnFunc() {
    auto func = getFunction();

    // TODO: move outside of pass
    StringRef fileNameOut("strategy_out.json");
    StringRef fileNameIn("strategy_in.json");

    bool writeStrategyToFile = true;
    bool readStrategyFromFile = true;

    // store operations with Location as key to enable Location based mapping
    llvm::DenseMap<mlir::Location, mlir::Operation*> operations;

    func->walk([&](IE::LayerOpInterface op) {
        operations.insert({op.getLoc(), op.getOperation()});
    });

    if (writeStrategyToFile) {
        // pass attributes name for creating JSON - filter
        SmallVector<StringRef> strategyAttributes = {"multiClusterStrategy"};

        // writing current strategy to json
        auto j = createStrategyJSONFromOperations(operations, strategyAttributes);
        writeManualStrategyJSON(fileNameOut, j);
    }

    if (readStrategyFromFile) {
        // reading strategy from json
        auto manualStrategy = readManualStrategyJSON(fileNameIn);

        // overwriting operation attributes
        if (!manualStrategy.is_null()) {
            overwriteManualStrategy(manualStrategy, operations);
        }
    }
}

// choose between reading/writing the IR
// if (!file_location) {
//     // save the filtered IR to file
//     auto generatedStrategy = func.collectManualStrategy();
//     writeASMTextFormatToFile(filename = passName, ASMText = generatedStrategy);
// } else {
//     // read and overwrite the IR from file
//     auto manualStrategy = readASMTextFormatToFile(filename = passName);
//     func.overwriteManualStrategy(manualStrategy);
// }

}  // namespace

//
// createManualStrategyUtilsPass
//

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(log);
}
