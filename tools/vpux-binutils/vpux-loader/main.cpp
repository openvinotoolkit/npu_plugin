//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/CommandLine.h>

#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_elf/utils/log.hpp>
#include <vpux_loader/vpux_loader.hpp>

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

class FlatHexBufferManager : public BufferManager {
public:
    FlatHexBufferManager(uint32_t startAddr, size_t size)
            : m_startAddr(startAddr), m_totalSize(size), m_buffer(new uint8_t[m_totalSize]), m_tracker(m_buffer) {
    }

    DeviceBuffer allocate(const BufferSpecs& buffSpecs) override {
        loaderLog.debug("Allocation request --> size {0} | alignment {1} | procFlags {2}", buffSpecs.size,
                        buffSpecs.alignment, buffSpecs.procFlags);

        if (!m_buffer) {
            loaderLog.error("Failed to allocate overall buffer of size {0}", buffSpecs.size);
            return elf::DeviceBuffer(nullptr, 0, 0);
        }

        m_tracker = align_up<uint8_t>(m_tracker, buffSpecs.alignment);
        uint8_t* start = m_tracker;
        m_tracker += buffSpecs.size;

        if (m_tracker >= m_buffer + m_totalSize) {
            loaderLog.error(
                    "Failed to allocate required buff of size {0} alignment {1} . Exceeding total device buffer space",
                    buffSpecs.size, buffSpecs.alignment);
            return elf::DeviceBuffer(nullptr, 0, 0);
        }

        return elf::DeviceBuffer(start, vpuWindow(start), buffSpecs.size);
    }

    void deallocate(elf::DeviceBuffer& devBuffer) override {
        (void)devBuffer;
    }

    ~FlatHexBufferManager() {
        delete[] m_buffer;
    }

    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) override {
        memcpy(to.cpu_addr(), from, count);
        return count;
    }

    uint8_t* buffer() const {
        return m_buffer;
    }
    size_t size() const {
        return m_tracker - m_buffer;
    }

private:
    template <typename T>
    bool isPowerOfTwo(T val) {
        return val && ((val & (val - 1)) == 0);
    }

    template <typename T>
    T* align_up(const T* val, const size_t to) {
        VPUX_ELF_THROW_UNLESS(isPowerOfTwo(to), " VPU only supports power of 2 alignments {0}", to);
        std::uintptr_t intPtr = reinterpret_cast<std::uintptr_t>(val);
        std::uintptr_t alignedAddr = (intPtr + to - 1) & ~(to - 1);
        return reinterpret_cast<T*>(alignedAddr);
    }

    uint32_t vpuWindow(uint8_t const* const addr) const {
        return m_startAddr + static_cast<uint32_t>(addr - m_buffer);
    }

    uint32_t const m_startAddr;
    size_t const m_totalSize;
    uint8_t* const m_buffer;
    uint8_t* m_tracker;
};

// TODO(E#23975): This beautiful piece of code contains the "special symtab" that normally needs to be queried from
// the runtime. In IMDemo example we have a similar class that constructs this symTab based on data from the
// InferenceRuntimeService. Since we cannot include that in kmb-plugin, we will occasionally manually check the values,
// and update them here Hopefully if we solve the riddle of integration between KmbPlugin and vpuip_2, then we will not
// have to resort to magical solutions (This is my wish to SantaClaus this year :) ) This ticket will not totally solve
// the problem, but will greatly reduce the hack-ishness of this solution

class HardCodedSymtabToCluster0 {
private:
    static constexpr size_t SPECIAL_SYMTAB_SIZE = 7;  // I counted!!!! Twice!!
    elf::SymbolEntry symTab_[SPECIAL_SYMTAB_SIZE];

public:
    HardCodedSymtabToCluster0(): symTab_() {
        for (size_t i = 0; i < SPECIAL_SYMTAB_SIZE; ++i) {
            symTab_[i].st_info = static_cast<unsigned char>(elf64STInfo(STB_GLOBAL, STT_OBJECT));
            symTab_[i].st_other = STV_DEFAULT;
            symTab_[i].st_shndx = 0;
            symTab_[i].st_name = 0;
        }

        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_value = 0x2e014000;
        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_size = 2097152;

        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_value = 0x2e004000;
        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_ACT].st_value = 0;
        symTab_[VPU_NNRD_SYM_RTM_ACT].st_size = 0;

        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_value = 0x2e1f8000;
        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_value = 0x2e1fc000;
        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_size = 64;

        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_value = 0x0;
        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_size = 0;

        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_value = 0;
        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_size = 0;
    }

    const details::ArrayRef<SymbolEntry> symTab() const {
        return details::ArrayRef<SymbolEntry>(symTab_, SPECIAL_SYMTAB_SIZE);
    }
};

void allocIO(BufferManager* mngr, std::vector<elf::DeviceBuffer>& ioVec, details::ArrayRef<DeviceBuffer> sizes,
             uint32_t* ioPtrs, uint32_t* ioSizesP, uint32_t* ioSize) {
    auto ioPtrBuf = mngr->allocate(BufferSpecs(1, sizes.size() * sizeof(uint32_t), SHF_NONE));
    auto ioSizesBuf = mngr->allocate(BufferSpecs(1, sizes.size() * sizeof(uint32_t), SHF_NONE));

    *ioPtrs = static_cast<uint32_t>(ioPtrBuf.vpu_addr());
    *ioSizesP = static_cast<uint32_t>(ioSizesBuf.vpu_addr());
    *ioSize = static_cast<uint32_t>(sizes.size());

    for (size_t i = 0; i < sizes.size(); ++i) {
        size_t size = sizes[i].size();
        auto buffer = mngr->allocate(BufferSpecs(1, size, SHF_NONE));

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

    HardCodedSymtabToCluster0 singleClusterSymTab;
    FlatHexBufferManager bufferManager(baseAddr, memSize);

    HexMappedInferenceEntry* hexEntry = reinterpret_cast<HexMappedInferenceEntry*>(
            bufferManager.allocate(BufferSpecs(1, sizeof(HexMappedInferenceEntry), SHF_NONE)).cpu_addr());

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
