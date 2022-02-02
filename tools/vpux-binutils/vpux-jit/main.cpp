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
#include <vpux_loader/vpux_loader.hpp>

#include "../common/binutils_common.hpp"
#include "simulator.hpp"

using namespace vpux::movisim;
using namespace elf;

namespace ie = InferenceEngine;

namespace {

constexpr uint32_t DEFAULT_BASE_BUFF = 0x80000000;
constexpr uint32_t DEFAULT_BUF_SIZE = 128 * 1024 * 1024;

enum InputSource { Random, Zeroes, Ones, CheckerBoard, Files };

// enum ExecutionMode {MLIR, ELF};

// llvm::cl::opt<std::string> inputFilePath(llvm::cl::Positional, llvm::cl::Required, llvm::cl::value_desc("filename"),
//                                         llvm::cl::desc("<Input MLIR File>"));

llvm::cl::OptionCategory executionMode("Execution mode", "These option control the execution mode of the program");

llvm::cl::opt<std::string> mlirFilePath("mlir",
                                    llvm::cl::desc("Use MLIR file as input. Run translation then loader then execute"),
                                    llvm::cl::cat(executionMode));
llvm::cl::opt<std::string> elfFilePath("elf", llvm::cl::desc("Use ELF file as input. Run the loader, then executer"),
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

llvm::cl::opt<uint32_t> baseAddr("addr", llvm::cl::value_desc("baseAddr"), llvm::cl::init(DEFAULT_BASE_BUFF),
                                 llvm::cl::desc("DEVELOPER ONLY: VPU base adress of the HEXFILE to be linked to."
                                                "This value needs to  reflect the InferenceManagerDemoHex"
                                                "application configuration"));

llvm::cl::opt<size_t> memSize("size", llvm::cl::value_desc("memSize"), llvm::cl::init(DEFAULT_BUF_SIZE),
                              llvm::cl::desc("DEVELOPER ONLY: Maximum size of the allocated buffer on VPU side"
                                             "for the Loaded and linked hex-file. This value needs to reflect"
                                             "the InferenceManagerDemoHex application configuration"));

llvm::cl::opt<bool> verboseL1("v", llvm::cl::desc("enable verbose execution"));
llvm::cl::opt<bool> verboseL2("vv", llvm::cl::desc("higher level of verbosity"), llvm::cl::Hidden);
llvm::cl::opt<bool> verboseL3("vvv", llvm::cl::desc("highest level of verbosity. Recommended for DEV only"),
                              llvm::cl::Hidden);

vpux::Logger appLogger("VPUX-JIT", vpux::LogLevel::Error);

void setLogLevel() {
    if (verboseL1) {
        appLogger.setLevel(vpux::LogLevel::Info);
    }
    if (verboseL2) {
        vpuxElfLogLevelSet(VPUX_ELF_DEBUG_LEVEL);
        appLogger.setLevel(vpux::LogLevel::Debug);
    }
    if (verboseL3) {
        vpuxElfLogLevelSet(VPUX_ELF_TRACE_LEVEL);
        appLogger.setLevel(vpux::LogLevel::Trace);
    }
    else{
        vpuxElfLogLevelSet(VPUX_ELF_WARN_LEVEL);
    }
}

void setInputs(std::vector<elf::DeviceBuffer>& vec) {
    if (inputSource == InputSource::Files) {
        if (vec.size() != inputFiles.size()) {
            appLogger.error("Elf has {0} required inputs. Only {1} specified", vec.size(), inputFiles.size());
            VPUX_ELF_THROW("Input Parameter mismatch");
        }
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        auto& buf = vec[i];
        if (inputSource == InputSource::Random) {
            std::srand(static_cast<unsigned int>(std::time(0)));
            auto rand = []() -> uint8_t {
                return static_cast<uint8_t>(std::rand() % std::numeric_limits<uint8_t>::max());
            };
            std::generate_n(buf.cpu_addr(), buf.size(), rand);
        } else if (inputSource == InputSource::Zeroes) {
            std::fill_n(buf.cpu_addr(), buf.size(), 0);
        } else if (inputSource == InputSource::Ones) {
            std::fill_n(buf.cpu_addr(), buf.size(), 1);
        } else if (inputSource == InputSource::CheckerBoard) {
            uint8_t flip = 0;
            auto flipper = [&]() -> uint8_t {
                flip = (flip + 1) % 2;
                return flip;
            };
            std::generate_n(buf.cpu_addr(), buf.size(), flipper);
        } else if (inputSource == InputSource::Files) {
            auto iFile = inputFiles[i];
            std::ifstream iFileStream(iFile.data(), std::ios::binary);
            iFileStream.read(reinterpret_cast<char*>(buf.cpu_addr()), buf.size());
            iFileStream.close();
        }
    }
}

}  // namespace

class IMDemoSimulator {
public:
    IMDemoSimulator(vpux::Logger logger)
            : m_sim("MTL_VPUX_JIT", Simulator::ARCH_MTL, logger),
              m_singleClusterSymTab(),
              m_bufferManager(baseAddr, memSize),
              m_entry(reinterpret_cast<vpux::binutils::HexMappedInferenceEntry*>(m_bufferManager.allocate(1, sizeof(vpux::binutils::HexMappedInferenceEntry)).cpu_addr())),
              m_logger(logger)

    {
        m_sim.start();
        m_sim.loadFile(ie::getIELibraryPath() + "/vpux_jit/VPUX_JIT_FW.elf");
    }

    void loadElf(void* elfFile, size_t elfSize) {
        VPUXLoader loader(elfFile, elfSize, m_singleClusterSymTab.symTab(), &m_bufferManager);

        m_inputSizes = loader.getInputBuffers().vec();
        m_outputSizes = loader.getOutputBuffers().vec();

        m_entry->elfEntryPtr = static_cast<uint32_t>(loader.getEntry());

        allocIO(&m_bufferManager, m_inputs, m_inputSizes, &m_entry->inputsPtr, &m_entry->inputSizesPtr, &m_entry->inputsCount);
        allocIO(&m_bufferManager, m_outputs, m_outputSizes, &m_entry->outputsPtr, &m_entry->outputSizesPtr, &m_entry->outputsCount);

        loader.applyJitRelocations(m_inputs, m_outputs);
    }

    std::vector<elf::DeviceBuffer>& getInputs() {
        return m_inputs;
    }

    std::vector<elf::DeviceBuffer>& getOutputs() {
        return m_outputs;
    }

    void readInto(uint32_t addr, std::vector<uint8_t>& targetVec) {
        m_sim.read(addr, targetVec.data(), targetVec.size());
    }

    void setInputs(std::vector<std::vector<uint8_t>>& inputs) {
        VPUX_THROW_WHEN(inputs.size() != m_inputs.size(), "Input count mismatch. Provided {0}. Required {1}", inputs.size(), m_inputs.size());
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            VPUX_THROW_WHEN(m_inputs[i].size() != input.size(), "Input size mismatch for input {0}. Provided {1}. Required {2}", i, input.size(), m_inputs[i].size());

            std::copy(input.begin(), input.end(), m_inputs[i].cpu_addr());
        }
    }

    void setOutputs(std::vector<std::vector<uint8_t>>& outputs) {
        VPUX_ELF_THROW_WHEN(outputs.size() != m_outputs.size(), "Output count mismatch. Provided {0}. Required {1}", outputs.size(), m_outputs.size());
        for(size_t i = 0; i < outputs.size(); ++i) {
            auto& output = outputs[i];
            VPUX_THROW_WHEN(output.size() != m_outputs.size(), "Output size mismatch for output {0}. Provided {1}. Required {2}", i, output.size(), m_outputs.size());

            std::copy(output.begin(), output.end(), m_outputs[i].cpu_addr());
        }
    }

    void run(/*mlir::ModuleOp module, llvm::ArrayRef<llvm::ArrayRef<uint8_t>> inputs,
             llvm::ArrayRef<llvm::ArrayRef<uint8_t>> outputs*/) {

        m_entry->totalSize = static_cast<uint32_t>(m_bufferManager.size());

        m_sim.write(m_bufferManager.vpuBaseAddr(), m_bufferManager.buffer(), m_bufferManager.size());
        m_sim.expectOutput("PIPE:\t: IMDEMOHEXT_FINISH\r\n");

        m_sim.resume();
        llvm::outs() << " 4 \n";

        llvm::outs() << " 5 \n";

        m_sim.resume();
        llvm::outs() << " 6 \n";

        m_sim.waitForExpectedOutputs(std::chrono::seconds(100));

        m_sim.reset();
        // while (1) {
        llvm::outs() << " FINISHED SIM \n";
        // };

        return;
    }

private:
    void allocIO(BufferManager* mngr, std::vector<DeviceBuffer>& ioVec, details::ArrayRef<DeviceBuffer> sizes,
                uint32_t* ioPtrs, uint32_t* ioSizesP, uint32_t* ioSize) {
        auto ioPtrBuf = mngr->allocate(1, sizes.size() * sizeof(uint32_t));
        auto ioSizesBuf = mngr->allocate(1, sizes.size() * sizeof(uint32_t));

        *ioPtrs = static_cast<uint32_t>(ioPtrBuf.vpu_addr());
        *ioSizesP = static_cast<uint32_t>(ioSizesBuf.vpu_addr());
        *ioSize = static_cast<uint32_t>(sizes.size());

        for (size_t i = 0; i < sizes.size(); ++i) {
            size_t size = sizes[i].size();
            auto buffer = mngr->allocate(1, size);

            ioVec.push_back(buffer);

            uint32_t* ioPtr = reinterpret_cast<uint32_t*>(ioPtrBuf.cpu_addr());
            uint32_t* ioSize = reinterpret_cast<uint32_t*>(ioSizesBuf.cpu_addr());

            ioPtr[i] = static_cast<uint32_t>(buffer.vpu_addr());
            ioSize[i] = static_cast<uint32_t>(buffer.size());
        }
    }

    Simulator m_sim;
    vpux::binutils::HardCodedSymtabToCluster0 m_singleClusterSymTab;
    vpux::binutils::FlatHexBufferManager m_bufferManager;
    vpux::binutils::HexMappedInferenceEntry* m_entry;

    std::vector<DeviceBuffer> m_inputs;
    std::vector<DeviceBuffer> m_outputs;
    std::vector<DeviceBuffer> m_inputSizes;
    std::vector<DeviceBuffer> m_outputSizes;

    vpux::Logger m_logger;
};

void dumpInputs(const std::vector<elf::DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Input count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());

        if(inputSource != InputSource::Files) {
            std::ofstream outFileStream;

            auto fileName = llvm::formatv("input-{0}.bin",i);
            outFileStream.open(fileName, std::ios::out | std::ios::binary);
            outFileStream.write(reinterpret_cast<const char*>(vec[i].cpu_addr()), vec[i].size());
            outFileStream.close();
        }

    }
}

void dumpOutputs(const std::vector<elf::DeviceBuffer>& vec, IMDemoSimulator& sim) {
    llvm::outs() << llvm::formatv("Output count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());

        std::ofstream outFileStream;
        std::vector<uint8_t> tempVec(vec[i].size());
        sim.readInto(vec[i].vpu_addr(), tempVec);

        auto fileName = llvm::formatv("output-{0}.bin",i);
        outFileStream.open(fileName, std::ios::out | std::ios::binary);
        outFileStream.write(reinterpret_cast<const char*>(tempVec.data()), tempVec.size());
        outFileStream.close();
    }
}

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
    setLogLevel();

    IMDemoSimulator appRunner(appLogger);

    if(mlirFilePath.size()) {
        appLogger.error("temporarily not supported :(");
        return 1;
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

    } else if(elfFilePath.size()) {
        std::ifstream inputStream(elfFilePath.data(), std::ios::binary);
        std::vector<uint8_t> elfFile((std::istreambuf_iterator<char>(inputStream)), (std::istreambuf_iterator<char>()));
        inputStream.close();

        appRunner.loadElf(elfFile.data(), elfFile.size());
    } else {
        appLogger.error("No input file specified");
        return 1;
    }

    setInputs(appRunner.getInputs());

    appRunner.run();

    dumpInputs(appRunner.getInputs());
    dumpOutputs(appRunner.getOutputs(), appRunner);

    return 0;
}
