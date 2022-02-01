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
#include <file_utils.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include <vpux/compiler/init.hpp>
#include <vpux/utils/core/logger.hpp>
#include <vpux_elf/reader.hpp>

#include "../common/binutils_common.hpp"
#include "simulator.hpp"

using namespace vpux::movisim;

namespace ie = InferenceEngine;

namespace {

enum InputSource { Random, Zeroes, Ones, CheckerBoard, Files };

// enum ExecutionMode {MLIR, ELF};

// llvm::cl::opt<std::string> inputFilePath(llvm::cl::Positional, llvm::cl::Required, llvm::cl::value_desc("filename"),
//                                         llvm::cl::desc("<Input MLIR File>"));

llvm::cl::OptionCategory executionMode("Execution mode", "These option control the execution mode of the program");

llvm::cl::opt<std::string> mlirFile("mlir",
                                    llvm::cl::desc("Use MLIR file as input. Run translation then loader then execute"),
                                    llvm::cl::cat(executionMode));
llvm::cl::opt<std::string> elfFile("elf", llvm::cl::desc("Use ELF file as input. Run the loader, then executer"),
                                   llvm::cl::cat(executionMode));

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

llvm::cl::opt<bool> verboseL1("v", llvm::cl::desc("enable verbose execution"));
llvm::cl::opt<bool> verboseL2("vv", llvm::cl::desc("higher level of verbosity"), llvm::cl::Hidden);
llvm::cl::opt<bool> verboseL3("vvv", llvm::cl::desc("highest level of verbosity. Recommended for DEV only"),
                              llvm::cl::Hidden);

vpux::Logger logger("VPUX-JIT", vpux::LogLevel::Error);

void setLogLevel() {
    if (verboseL1) {
        logger.setLevel(vpux::LogLevel::Info);
    }
    if (verboseL2) {
        logger.setLevel(vpux::LogLevel::Debug);
    }
    if (verboseL3) {
        logger.setLevel(vpux::LogLevel::Trace);
    }
}

}  // namespace

class IMDemoSimulator {
public:

    IMDemoSimulator(vpux::Logger logger)
        : m_sim("MTL_VPUX_JIT", Simulator::ARCH_MTL, logger),

    {

    }
    void loadElf(elf::Reader& elf) {

    }

    void run(/*mlir::ModuleOp module, llvm::ArrayRef<llvm::ArrayRef<uint8_t>> inputs,
             llvm::ArrayRef<llvm::ArrayRef<uint8_t>> outputs*/) {
        // llvm::outs() << " 1 \n";
        // Simulator::get().setLogLevel(movisim::MovisimLogLevel::MOVISIM_ALL);
        Simulator sim("MTL_Test", Simulator::ARCH_MTL, logger);
        llvm::outs() << " 2 \n";
        sim.start();
        llvm::outs() << " 22\n";

        sim.loadFile(ie::getIELibraryPath() + "/vpux_jit/VPUX_JIT_FW.elf");
        llvm::outs() << " 3 " << ie::getIELibraryPath() + "/vpux_jit/VPUX_JIT_FW.elf \n";

        sim.resume();
        llvm::outs() << " 4 \n";

        sim.expectOutput("PIPE:\t: TestLeonFinish\r\n");
        llvm::outs() << " 5 \n";

        sim.resume();
        llvm::outs() << " 6 \n";

        sim.waitForExpectedOutputs(std::chrono::seconds(100));

        sim.reset();
        // while (1) {
        llvm::outs() << " FINISHED SIM \n";
        // };

        return;
    }

private:
    Simulator m_sim;
    vpux::binutils::HardCodedSymtabToCluster0 m_singleClusterSymTab;
    vpux::binutils::FlatHexBufferManager m_bufferManager;
};

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
    setLogLevel();

    // std::string errorMessage;
    // auto mlirFile = mlir::openInputFile(mlirFilePath, &errorMessage);
    // if (!mlirFile) {
    //     logger.error("{0}", errorMessage);
    // }
    // mlir::MLIRContext context;
    // mlir::DialectRegistry dialectRegistry;
    // vpux::registerDialects(dialectRegistry);
    // context.appendDialectRegistry(dialectRegistry);

    // llvm::SourceMgr sourceMgr;
    // sourceMgr.AddNewSourceBuffer(std::move(mlirFile), llvm::SMLoc());

    // auto module = mlir::OwningModuleRef(mlir::parseSourceFile(sourceMgr, &context));

    // std::vector<std::vector<uint8_t>> inputs;
    // std::vector<std::vector<uint8_t>> outputs;
    IMDemoSimulator sim;
    // sim.run(module, inputs, outputs);
    sim.run();
    // call the translation, generate the elf yadayada

    // std::array<char*, 3> args {
    //     nullptr,
    //     const_cast<char*>("-cv:3720xx"),
    //     const_cast<char*>("-nodasm")
    // };

    // typedef movisim::MovisimInterface* (*createMoviSimObjectPtr)();
}
