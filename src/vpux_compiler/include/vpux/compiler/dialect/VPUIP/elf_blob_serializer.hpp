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

#pragma once

#include <elf/writer.hpp>
#include <elf/reader32.hpp>

#include <vpux/compiler/dialect/VPUIP/attributes/enums.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <host_parsed_inference.h>

#include <map>

namespace vpux {
namespace VPUIP {

struct TensorLocation {
    VPUIP::MemoryLocation memLocation;
    uint64_t locationIndex = 0;
};

enum class MetaDataLocation { NONE = 0, RTM_INV, RTM_DMA, DDR_DMA };

struct TensorPatchingInfo {
    TensorLocation location;
    uint64_t offset = 0;
};

struct DMATaskExtension {
    TensorPatchingInfo input;
    TensorPatchingInfo output;
};

class ELFBlobSerializer {
public:
    ELFBlobSerializer();

    void initActKernel(std::vector<char> elfBlob, std::string name);
    void addActKernel();
    void addActInvocation();
    void finalizeActKernelWrappers();

    void setNetworkInputs(llvm::ArrayRef<mlir::MemRefType> inputs);
    void setNetworkOutputs(llvm::ArrayRef<mlir::MemRefType> outputs);

    void setDDRScratch(size_t ddrScratch);
    void setResourceRequirements(const ResourceRequirements& resourceRequirements);

    void setLeadingDMACount0(uint32_t leadingDMACount);
    void setDMATasks0(llvm::ArrayRef<std::pair<DmaWrapper, DMATaskExtension>> dmaTasks);
    void setBarrierConfigs(llvm::ArrayRef<BarrierWrapper> barrierConfigs);

    std::vector<char> getBlob();
    void write(const std::string& fileName);

private:
    elf::writer::Symbol* getSymbol(const TensorLocation& location);
    elf::writer::Symbol* getSymbol(VPUIP::MemoryLocation location);
    elf::writer::SymbolSection* getSymbolSection(const TensorLocation& location);

    elf::writer::RelocationSection* createRelocationSection(VPUIP::MemoryLocation location);

    void setNetworkIO(llvm::ArrayRef<mlir::MemRefType> inputsOrOutputs, uint8_t symbolType,
                      elf::writer::SymbolSection*& symbolSection, const std::string& symbolName);
    void finalize();

private:
    elf::Writer m_writer;

    size_t m_ddrScratch;
    ResourceRequirements m_resourceRequirements;
    MappedInference m_mappedInference;

    elf::writer::SymbolSection* m_networkInputSymbols = nullptr;
    elf::writer::SymbolSection* m_networkOutputSymbols = nullptr;

    elf::writer::SymbolSection* m_symbols = nullptr;
    std::map<VPUIP::MemoryLocation, elf::writer::Symbol*> m_symbolsMapping;

    elf::writer::BinaryDataSection<DmaWrapper>* m_dmaTasks0 = nullptr;
    elf::writer::BinaryDataSection<BarrierWrapper>* m_barrierConfigs = nullptr;

    // ALL ABOUT KERNELS
    elf::Reader32 m_reader;
    bool isReaderInit = false;

    int m_kernelsNum = 0;
    std::string m_kernelName = "actKernel";
    elf::writer::SymbolSection* m_actKernel_symbols = nullptr;
    std::map<std::string, std::vector<std::string>> m_actKernelsMapping;

    // Act Kernel Ranges
    elf::writer::Symbol* m_actKernelRangeSymbol = nullptr;
    elf::writer::BinaryDataSection<ActKernelRange>* m_temp_actKernelRanges = nullptr;
    elf::writer::RelocationSection* m_temp_actKernelRangeRela = nullptr;
    elf::writer::BinaryDataSection<ActKernelRangeWrapper>* m_actKernelRanges = nullptr;
    elf::writer::RelocationSection* m_actKernelRangeRela = nullptr;

    // Act Kernel Invocations
    elf::writer::Symbol* m_actKernelInvocationSymbol = nullptr;
    elf::writer::BinaryDataSection<ActKernelInvocation>* m_temp_actKernelInvocations = nullptr;
    elf::writer::RelocationSection* m_temp_actKernelInvocationRela = nullptr;
    elf::writer::BinaryDataSection<ActKernelInvocationWrapper>* m_actKernelInvocations = nullptr;
    elf::writer::RelocationSection* m_actKernelInvocationRela = nullptr;
};

}  // namespace VPUIP
}  // namespace vpux
