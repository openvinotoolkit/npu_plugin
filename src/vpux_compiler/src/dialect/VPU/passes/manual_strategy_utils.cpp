//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
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
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnFunc() final;

private:
    bool _writeStrategyToJSON;
    StringRef _writeStrategyFileLocation;
    bool _readStrategyFromJSON;
    StringRef _readStrategyFileLocation;
};

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 Logger log)
        // NOTE: currently called after two strategy passes, flags in both must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult ManualStrategyUtilsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    if (writeStrategyToJSON.hasValue()) {
        _writeStrategyToJSON = writeStrategyToJSON.getValue();
    }

    if (writeStrategyFileLocation.hasValue()) {
        _writeStrategyFileLocation = writeStrategyFileLocation.getValue();
    }

    if (readStrategyFromJSON.hasValue()) {
        _readStrategyFromJSON = readStrategyFromJSON.getValue();
    }

    if (readStrategyFileLocation.hasValue()) {
        _readStrategyFileLocation = readStrategyFileLocation.getValue();
    }

    return mlir::success();
}

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

StringRef parseEnv(StringRef envVarName, StringRef var) {
    if (const auto env = std::getenv(envVarName.data())) {
        // update if new location specified
        return StringRef(env);
    } else {
        return var;
    }
}

void parseEnv(StringRef envVarName, bool& var) {
    if (const auto env = std::getenv(envVarName.data())) {
        var = std::stoi(env);
    }
}

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

//
// safeRunOnFunc
//

void ManualStrategyUtilsPass::safeRunOnFunc() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_VPUX_WRITE_STRATEGY_JSON", _writeStrategyToJSON);
    _writeStrategyFileLocation = parseEnv("IE_VPUX_WRITE_STRATEGY_JSON_LOC", _writeStrategyFileLocation);
    parseEnv("IE_VPUX_READ_STRATEGY_JSON", _readStrategyFromJSON);
    _readStrategyFileLocation = parseEnv("IE_VPUX_READ_STRATEGY_JSON_LOC", _readStrategyFileLocation);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    auto func = getFunction();

    if (!_writeStrategyToJSON && !_readStrategyFromJSON) {
        _log.trace("Flags to write and read disabled, skipping pass");
        return;
    }

    if (_readStrategyFromJSON && _readStrategyFileLocation.empty()) {
        _log.error("Invalid read location for manual strategy");
        signalPassFailure();
        return;
    }

    if (_writeStrategyToJSON && _writeStrategyFileLocation.empty()) {
        _log.error("Invalid write location for manual strategy");
        signalPassFailure();
        return;
    }

    _log.trace("Starting Manual Strategy Pass");
    _log.nest(1).trace("Option to write strategy: '{0}'", _writeStrategyToJSON);
    _log.nest(1).trace("Strategy write file location: '{0}'", _writeStrategyFileLocation);
    _log.nest(1).trace("Option to read strategy: '{0}'", _readStrategyFromJSON);
    _log.nest(1).trace("Strategy read file location: '{0}'", _readStrategyFileLocation);

    // store operations with Location as key to enable Location based mapping
    llvm::MapVector<mlir::Location, mlir::Operation*> operations;

    bool operationsWrappedInClusterTiling = false;
    bool operationsHaveTilingAttr = false;

    func->walk([&](VPU::NCEOpInterface op) {
        // store unique operations (tiled operations are merged)
        mlir::Location opLoc = nullptr;
        if (op->hasAttr(tilingStrategy)) {
            const auto fused = op.getLoc().dyn_cast<mlir::FusedLoc>();
            VPUX_THROW_UNLESS(fused, "Tiled operation has location not fused");
            // store only 1 tile
            opLoc = fused.getLocations().front();
        } else {
            opLoc = op.getLoc();
            if (operations.find(opLoc) != operations.end()) {
                // if duplicate locations, create unique
                opLoc = appendLoc(opLoc, "unique_{0}", operations.count(opLoc));
                op->setLoc(opLoc);
            }
        }
        operations.insert({opLoc, op.getOperation()});
        if (!operationsWrappedInClusterTiling && op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
            _log.nest(2).trace("Operations wrapped in cluster tiling exist");
            operationsWrappedInClusterTiling = true;
        }
        if (!operationsHaveTilingAttr && op->hasAttr(tilingStrategy)) {
            _log.nest(2).trace("Tiled operations exist");
            operationsHaveTilingAttr = true;
        }
    });

    if (_writeStrategyToJSON) {
        _log.nest(1).trace("Writing strategy to JSON");
        // pass attributes name for creating JSON - filter
        // currently supported attributes
        //  - multiClusterStrategy
        //  - tilingStrategy
        SmallVector<StringRef> strategyAttributes = {multiClusterStrategy, tilingStrategy};

        Json json;
        if (operationsWrappedInClusterTiling) {
            // read stategies from first strategy pass and append new strategies
            _log.nest(2).trace("Appending to strategies from first strategy pass");
            json = readManualStrategyJSON(_writeStrategyFileLocation);
        }
        // writing current strategy to json
        createStrategyJSONFromOperations(json, operations, strategyAttributes);
        writeManualStrategyJSON(_writeStrategyFileLocation, json);
    }

    if (_readStrategyFromJSON) {
        if (!operationsWrappedInClusterTiling && !operationsHaveTilingAttr) {
            _log.nest(1).trace("Reading strategy from JSON");
            // reading strategy from json only during first pass call
            auto manualStrategy = readManualStrategyJSON(_readStrategyFileLocation);

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

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                               StringRef writeStrategyFileLocation,
                                                               bool readStrategyFromJSON,
                                                               StringRef readStrategyFileLocation, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, writeStrategyFileLocation,
                                                     readStrategyFromJSON, readStrategyFileLocation, log);
}
