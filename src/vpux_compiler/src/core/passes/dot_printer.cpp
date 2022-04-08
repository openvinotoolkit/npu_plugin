//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/dot_printer.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/utils/dot_graph_writer.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/Support/Regex.h>

#include <iostream>
#include <memory>
#include <string>

using namespace vpux;

namespace {

//
// PrintDotPass
//

class PrintDotPass final : public PrintDotBase<PrintDotPass> {
public:
    PrintDotPass() = default;
    PrintDotPass(StringRef fileName, const GraphWriterParams& writerParams)
            : _fileName(fileName.str()), _writerParams(writerParams) {
    }

public:
    mlir::LogicalResult initializeOptions(StringRef options) final;

public:
    void processModule(mlir::ModuleOp module);
    void processFunc(mlir::FuncOp func);

public:
    bool checkPass(mlir::Pass* pass) const;

private:
    void safeRunOnModule() final {
        processModule(getOperation());
    }

private:
    std::string _fileName;
    GraphWriterParams _writerParams;
    std::shared_ptr<llvm::Regex> _passNameFilter;
};

mlir::LogicalResult PrintDotPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    if (printDeclarationsOpt.hasValue()) {
        _writerParams.printDeclarations = printDeclarationsOpt.getValue();
    }
    if (printConstOpt.hasValue()) {
        _writerParams.printConst = printConstOpt.getValue();
    }
    if (startAfterOpt.hasValue()) {
        _writerParams.startAfter = startAfterOpt.getValue();
    }
    if (stopBeforeOpt.hasValue()) {
        _writerParams.stopBefore = stopBeforeOpt.getValue();
    }
    if (outputFileOpt.hasValue()) {
        _fileName = outputFileOpt.getValue();
    }
    if (afterPassOpt.hasValue()) {
        _passNameFilter = std::make_shared<llvm::Regex>(afterPassOpt.getValue(), llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_passNameFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", afterPassOpt.getValue(), regexErr);
        }
    }

    return mlir::success();
}

bool PrintDotPass::checkPass(mlir::Pass* pass) const {
    VPUX_THROW_WHEN(_passNameFilter == nullptr, "Pass name filter was not specified");
    return _passNameFilter->match(pass->getName()) || _passNameFilter->match(pass->getArgument());
}

void PrintDotPass::processModule(mlir::ModuleOp module) {
    module.walk([this](mlir::FuncOp func) {
        processFunc(func);
    });
}

void PrintDotPass::processFunc(mlir::FuncOp func) {
    VPUX_THROW_WHEN(_fileName.empty(), "Output file name for PrintDot was not provided");

    for (auto& block : func.getBody()) {
        VPUX_THROW_WHEN(mlir::failed(writeDotGraph(block, _fileName, _writerParams)), "Could not create Dot File");
    }
}

//
// PrintDotInstrumentation
//

class PrintDotInstrumentation final : public mlir::PassInstrumentation {
public:
    void addPrinter(std::unique_ptr<PrintDotPass> printer) {
        _printers.push_back(std::move(printer));
    }

public:
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op);

private:
    std::vector<std::unique_ptr<PrintDotPass>> _printers;
};

void PrintDotInstrumentation::runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
    for (auto& printer : _printers) {
        if (!printer->checkPass(pass)) {
            continue;
        }

        llvm::outs() << " Generating Dot after " << pass->getName() << "\n";

        if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) {
            printer->processModule(module);
        } else if (auto func = mlir::dyn_cast<mlir::FuncOp>(op)) {
            printer->processFunc(func);
        }
    }
}

}  // namespace

//
// addDotPrinter
//

void vpux::addDotPrinter(mlir::PassManager& pm, StringRef options) {
    auto instr = std::make_unique<PrintDotInstrumentation>();

    std::stringstream ss(options.data());
    std::string split;
    while (std::getline(ss, split, ',')) {
        std::unique_ptr<PrintDotPass> printer = std::make_unique<PrintDotPass>();
        VPUX_THROW_WHEN(mlir::failed(printer->initializeOptions(split)), "Failed to initialize options");

        instr->addPrinter(std::move(printer));
    }

    pm.addInstrumentation(std::move(instr));
}

//
// createPrintDotPass
//

std::unique_ptr<mlir::Pass> vpux::createPrintDotPass(StringRef fileName, StringRef startAfter, StringRef stopBefore,
                                                     bool printConst, bool printDeclarations) {
    GraphWriterParams writerParams;
    writerParams.printConst = printConst;
    writerParams.printDeclarations = printDeclarations;
    writerParams.startAfter = startAfter.str();
    writerParams.stopBefore = stopBefore.str();

    return std::make_unique<PrintDotPass>(fileName, writerParams);
}
