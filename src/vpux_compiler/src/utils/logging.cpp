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

#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

using namespace vpux;

//
// Context logging
//

void vpux::addLogging(mlir::MLIRContext& ctx, Logger log) {
    auto& diagEngine = ctx.getDiagEngine();

    diagEngine.registerHandler([log](mlir::Diagnostic& diag) -> mlir::LogicalResult {
        const auto severity = diag.getSeverity();
        const auto msgLevel = [severity]() -> LogLevel {
            switch (severity) {
            case mlir::DiagnosticSeverity::Note:
            case mlir::DiagnosticSeverity::Remark:
                return LogLevel::Info;

            case mlir::DiagnosticSeverity::Warning:
                return LogLevel::Warning;

            case mlir::DiagnosticSeverity::Error:
                return LogLevel::Error;
            default:
                return LogLevel::None;
            }
        }();

        const auto loc = diag.getLocation();
        log.addEntry(msgLevel, "Got Diagnostic at {0} : {1}", loc, diag);

        // Propagate diagnostic to following handlers
        return mlir::failure();
    });
}

//
// PassLogging
//

namespace {

class PassLogging final : public mlir::PassInstrumentation {
public:
    explicit PassLogging(Logger log): _log(log) {
    }

    void runBeforePipeline(mlir::Identifier name, const PipelineParentInfo&) final {
        _log.trace("Start Pass Pipeline {0}", name);
    }

    void runAfterPipeline(mlir::Identifier name, const PipelineParentInfo&) final {
        _log.trace("End Pass Pipeline {0}", name);
    }

    void runBeforePass(mlir::Pass* pass, mlir::Operation* op) final {
        _log.trace("Start Pass {0} on Operation {1}", pass->getName(), op->getLoc());
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
        _log.trace("End Pass {0} on Operation {1}", pass->getName(), op->getLoc());
    }

    void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) {
        _log.error("Failed Pass {0} on Operation {1}", pass->getName(), op->getLoc());
    }

    void runBeforeAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) {
        _log.trace("Start Analysis {0} on Operation {1}", name, op->getLoc());
    }

    void runAfterAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) {
        _log.trace("End Analysis {0} on Operation {1}", name, op->getLoc());
    }

private:
    Logger _log;
};

}  // namespace

void vpux::addLogging(mlir::PassManager& pm, Logger log) {
    pm.addInstrumentation(std::make_unique<PassLogging>(log));
}

//
// OpBuilderLogger
//

void vpux::OpBuilderLogger::notifyOperationInserted(mlir::Operation* op) {
    _log.trace("Add new Operation {0}", op->getLoc());
}

void vpux::OpBuilderLogger::notifyBlockCreated(mlir::Block* block) {
    if (auto* parent = block->getParentOp()) {
        _log.trace("Add new Block for Operation {0}", parent->getLoc());
    } else {
        _log.trace("Add new Block without parent Operation");
    }
}
