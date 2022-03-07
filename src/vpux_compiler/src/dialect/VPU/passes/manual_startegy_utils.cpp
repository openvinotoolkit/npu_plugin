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
    ManualStrategyUtilsPass() = default;
    ManualStrategyUtilsPass(bool writeStrategyToJSON, bool readStrategyFromJSON, StringRef strategyFileLocation,
                            Logger log);

private:
    void safeRunOnFunc() final;

private:
    bool _writeStrategyToJSON;
    bool _readStrategyFromJSON;
    StringRef _strategyFileLocation;
};

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, bool readStrategyFromJSON,
                                                 StringRef strategyFileLocation, Logger log)
        // NOTE: currently called after two strategy passes, flags in both must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _readStrategyFromJSON(readStrategyFromJSON),
          _strategyFileLocation(strategyFileLocation) {
    Base::initLogger(log, Base::getArgumentName());
}

//
// safeRunOnFunc
//

void ManualStrategyUtilsPass::safeRunOnFunc() {
    auto func = getFunction();

    if (!_writeStrategyToJSON && !_readStrategyFromJSON) {
        _log.trace("Flags to write and read disabled, skipping pass");
        return;
    }

    if (_strategyFileLocation.empty()) {
        _log.trace("Invalid location for manual strategy, skipping pass");
        return;
    }

    _log.trace("Starting Manual Strategy Pass");
    _log.nest(1).trace("Option to write strategy: '{0}'", _writeStrategyToJSON);
    _log.nest(1).trace("Option to read strategy: '{0}'", _readStrategyFromJSON);
    _log.nest(1).trace("Strategy file location: '{0}'", _strategyFileLocation);

    // store operations with Location as key to enable Location based mapping
    llvm::DenseMap<mlir::Location, mlir::Operation*> operations;

    bool operationsWrappedInClusterTiling = false;
    bool operationsHaveTilingAttr = false;

    func->walk([&](IE::LayerOpInterface op) {
        if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
            mlir::isa<VPU::NCEEltwiseOp>(op) || mlir::isa<VPU::NCEMaxPoolOp>(op)) {
            // store unique operations (tiled operations are merged)
            if (const auto fused = op.getLoc().dyn_cast<mlir::FusedLoc>()) {
                operations.insert({fused.getLocations().front(), op.getOperation()});
            } else {
                operations.insert({op.getLoc(), op.getOperation()});
            }
            if (!operationsWrappedInClusterTiling && op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
                _log.nest(2).trace("Operations wrapped in cluster tiling exist");
                operationsWrappedInClusterTiling = true;
            }
            if (!operationsHaveTilingAttr && op->hasAttr("tilingStrategy")) {
                _log.nest(2).trace("Tiled operations exist");
                operationsHaveTilingAttr = true;
            }
        }
    });

    if (_writeStrategyToJSON) {
        _log.nest(1).trace("Writing strategy to JSON");
        // pass attributes name for creating JSON - filter
        // currently supported attributes
        //  - multiClusterStrategy
        //  - tilingStrategy
        SmallVector<StringRef> strategyAttributes = {"multiClusterStrategy", "tilingStrategy"};

        Json j;
        if (operationsWrappedInClusterTiling) {
            // read stategies from first strategy pass and append new strategies
            _log.nest(2).trace("Appending to strategies from first strategy pass");
            j = readManualStrategyJSON(_strategyFileLocation);
        }
        // writing current strategy to json
        j = createStrategyJSONFromOperations(j, operations, strategyAttributes);
        writeManualStrategyJSON(_strategyFileLocation, j);
    }

    if (_readStrategyFromJSON) {
        _log.nest(1).trace("Reading strategy from JSON");
        if (!operationsHaveTilingAttr) {
            // reading strategy from json only during first pass call
            auto manualStrategy = readManualStrategyJSON(_strategyFileLocation);

            // overwriting operation attributes
            if (!manualStrategy.is_null()) {
                Logger::global().warning("WARNING: Experimental mode - assigning manual strategies");
                overwriteManualStrategy(manualStrategy, operations);
            }
        }
    }
}

}  // namespace

//
// createManualStrategyUtilsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createManualStrategyUtilsPass() {
    return std::make_unique<ManualStrategyUtilsPass>();
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON, bool readStrategyFromJSON,
                                                               StringRef strategyFileLocation, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, readStrategyFromJSON, strategyFileLocation,
                                                     log);
}
