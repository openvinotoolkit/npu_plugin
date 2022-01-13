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

#include <mlir/Support/FileUtilities.h>

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

} // namespace

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

}
