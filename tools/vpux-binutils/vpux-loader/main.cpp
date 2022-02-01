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

#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_elf/utils/log.hpp>
#include <vpux_loader/vpux_loader.hpp>

#include "../common/binutils_common.hpp"

#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace elf;

namespace {

constexpr uint32_t DEFAULT_BASE_BUFF = 0x80000000;
constexpr uint32_t DEFAULT_BUF_SIZE = 128 * 1024 * 1024;

enum InputSource { Random, Zeroes, Ones, CheckerBoard, Files };

llvm::cl::opt<std::string> elfFilePath(llvm::cl::Positional, llvm::cl::Required, llvm::cl::value_desc("filename"),
                                       llvm::cl::init("vpu.elf"), llvm::cl::desc("<Input ELF file>"));

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

llvm::cl::opt<uint32_t> baseAddr("addr", llvm::cl::value_desc("baseAddr"), llvm::cl::init(DEFAULT_BASE_BUFF),
                                 llvm::cl::desc("DEVELOPER ONLY: VPU base adress of the HEXFILE to be linked to."
                                                "This value needs to  reflect the InferenceManagerDemoHex"
                                                "application configuration"));

llvm::cl::opt<size_t> memSize("size", llvm::cl::value_desc("memSize"), llvm::cl::init(DEFAULT_BUF_SIZE),
                              llvm::cl::desc("DEVELOPER ONLY: Maximum size of the allocated buffer on VPU side"
                                             "for the Loaded and linked hex-file. This value needs to reflect"
                                             "the InferenceManagerDemoHex application configuration"));

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Set verbosity"));

vpux::Logger loaderLog("VPUX-LOADER", vpux::LogLevel::Warning);

}  // namespace

void allocIO(BufferManager* mngr, std::vector<elf::DeviceBuffer>& ioVec, details::ArrayRef<DeviceBuffer> sizes,
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

void setInputs(std::vector<elf::DeviceBuffer>& vec) {
    if (inputSource == InputSource::Files) {
        if (vec.size() != inputFiles.size()) {
            loaderLog.error("Elf has {0} required inputs. Only {1} specified", vec.size(), inputFiles.size());
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

void dumpOutputs(const std::vector<elf::DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Output count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());
    }
}

void dumpInputs(const std::vector<elf::DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Input count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());
    }
}

struct HexMappedInferenceEntry {
    uint32_t elfEntryPtr;
    uint32_t totalSize;
    uint32_t inputsPtr;
    uint32_t inputSizesPtr;
    uint32_t inputsCount;
    uint32_t outputsPtr;
    uint32_t outputSizesPtr;
    uint32_t outputsCount;
};

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    if (verbose) {
        vpuxElfLogLevelSet(VPUX_ELF_DEBUG_LEVEL);
        loaderLog.setLevel(vpux::LogLevel::Debug);

    } else {
        vpuxElfLogLevelSet(VPUX_ELF_WARN_LEVEL);
        loaderLog.setLevel(vpux::LogLevel::Warning);
    }

    std::ifstream inputStream(elfFilePath.data(), std::ios::binary);
    std::vector<uint8_t> elfFile((std::istreambuf_iterator<char>(inputStream)), (std::istreambuf_iterator<char>()));
    inputStream.close();

    vpux::binutils::HardCodedSymtabToCluster0 singleClusterSymTab;
    vpux::binutils::FlatHexBufferManager bufferManager(baseAddr, memSize);

    HexMappedInferenceEntry* hexEntry = reinterpret_cast<HexMappedInferenceEntry*>(
            bufferManager.allocate(1, sizeof(HexMappedInferenceEntry)).cpu_addr());

    elf::VPUXLoader loader(reinterpret_cast<void*>(elfFile.data()), elfFile.size(), singleClusterSymTab.symTab(),
                           &bufferManager);

    std::vector<DeviceBuffer> inputs;
    std::vector<DeviceBuffer> outputs;
    auto inputSizes = loader.getInputBuffers();
    auto outputSizes = loader.getOutputBuffers();

    allocIO(&bufferManager, inputs, inputSizes, &hexEntry->inputsPtr, &hexEntry->inputSizesPtr, &hexEntry->inputsCount);
    allocIO(&bufferManager, outputs, outputSizes, &hexEntry->outputsPtr, &hexEntry->outputSizesPtr,
            &hexEntry->outputsCount);
    hexEntry->elfEntryPtr = static_cast<uint32_t>(loader.getEntry());

    setInputs(inputs);

    llvm::outs() << llvm::formatv("Mapped Inferece pointer at:\n\t {0:x}\n", loader.getEntry());
    dumpInputs(inputs);
    dumpOutputs(outputs);

    loader.applyJitRelocations(inputs, outputs);

    hexEntry->totalSize = static_cast<uint32_t>(bufferManager.size());

    std::ofstream outFileStream;

    outFileStream.open(outputFile.data(), std::ios::out | std::ios::binary);
    outFileStream.write(reinterpret_cast<char*>(bufferManager.buffer()), bufferManager.size());
    outFileStream.close();

    VPUX_ELF_THROW_UNLESS(outFileStream.good(), "Error at writing the output file");

    return 0;
}
