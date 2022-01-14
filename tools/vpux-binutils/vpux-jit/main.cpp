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

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include <vpux/compiler/init.hpp>
#include "vpux/utils/core/logger.hpp"

namespace {

enum InputSource { Random, Zeroes, Ones, CheckerBoard, Files };

llvm::cl::opt<std::string> mlirFilePath(llvm::cl::Positional, llvm::cl::Required, llvm::cl::value_desc("filename"),
                                        llvm::cl::desc("<Input MLIR File>"));

llvm::cl::opt<InputSource> inputSource(
        "i",
        llvm::cl::values(clEnumValN(Random, "random", "Random number generation"),
                         clEnumValN(Zeroes, "zeroes", "Fill with Zeores"), clEnumValN(Ones, "ones", "Fill ones"),
                         clEnumValN(CheckerBoard, "checkerBoard", "Fill with checkerBoardPattern"),
                         clEnumValN(Files, "files", "FileList specified by -f commandLine list")),
        llvm::cl::init(Files), llvm::cl::desc("Input buffer source: "));

llvm::cl::list<std::string> inputFiles("f", llvm::cl::value_desc("Input Files"),
                                       llvm::cl::desc("Input file names. If buffer source is set to a generator "
                                                      "then this is ignored"));

llvm::cl::opt<std::string> outputFile("o", llvm::cl::value_desc("filename"), llvm::cl::init("vpu.bin"),
                                      llvm::cl::desc("Output file name"));

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("enable verbose execution"));

vpux::Logger logger("VPUX-JIT", vpux::LogLevel::Warning);

}  // namespace


int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    std::string errorMessage;
    auto mlirFile = mlir::openInputFile(mlirFilePath, &errorMessage);
    if (!mlirFile) {
        logger.error("{0}", errorMessage);
    }

    mlir::MLIRContext context;
    mlir::DialectRegistry dialectRegistry;
    vpux::registerDialects(dialectRegistry);
    context.appendDialectRegistry(dialectRegistry);

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(mlirFile), llvm::SMLoc());

    auto module = mlir::OwningModuleRef(mlir::parseSourceFile(sourceMgr, &context));



}
