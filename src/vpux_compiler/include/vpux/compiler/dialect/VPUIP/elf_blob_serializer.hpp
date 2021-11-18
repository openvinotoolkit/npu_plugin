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
#include <unordered_map>

namespace vpux {
namespace VPUIP {

struct TensorLocation {
    VPUIP::MemoryLocation memLocation;
    uint64_t locationIndex = 0;
};

struct TensorPatchingInfo {
    TensorLocation location;
    uint64_t dataOffset = 0;
    uint64_t sparsityMapOffset = 0;
    uint64_t sparsityTableOffset = 0;
};

enum class OffsetToUse { BASE = 0, DATA = 0, SPARSITY_MAP, SPARSITY_TABLE };

//
// DMA
//

struct LinkAddressPatchingInfo {
    enum class MetaDataLocation { NONE = 0, DDR_DMA, RTM_DMA } metaDataLocation = MetaDataLocation::NONE;
    size_t dmaTaskIndex = 0;
};

struct DmaTask {
    host_parsing::DmaWrapper dmaDescriptor{};
    TensorPatchingInfo input{};
    TensorPatchingInfo output{};
    LinkAddressPatchingInfo linkAddress{};
};

//
// DPU
//

struct DPUInvariantTask {
    host_parsing::DPUInvariantWrapper dpuInvariantWrapper{};
    TensorPatchingInfo input{};
    TensorPatchingInfo weights{};
    TensorPatchingInfo weightsTable{};
    TensorPatchingInfo output{};
    vpux::VPUIP::NCETaskType opType;
};

struct DPUVariantTask {
    host_parsing::DPUVariantWrapper dpuVariantWrapper;
};

struct DPUTask {
    DPUInvariantTask dpuInvariant;
    std::vector<DPUVariantTask> dpuVariants;
};

constexpr uint32_t SLICE_LENGTH = 1024 * 1024;

class ELFBlobSerializer {
public:
    ELFBlobSerializer();

    void setNetworkInputs(llvm::ArrayRef<mlir::ShapedType> inputs);
    void setNetworkOutputs(llvm::ArrayRef<mlir::ShapedType> outputs);
    
    void initActKernel(std::vector<char> elfBlob, std::string name);
    void addActKernel();
    void addActInvocation();
    void finalizeActKernelWrappers();

    void setDDRScratch(size_t ddrScratch);
    void setResourceRequirements(const host_parsing::ResourceRequirements& resourceRequirements);

    void setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex = 0);
    void setDMATasks(llvm::ArrayRef<DmaTask> dmaTasks, size_t dmaEngineIndex = 0);
    void setDPUTasks(llvm::ArrayRef<DPUTask> dpuTasks);
    void setBarrierConfigs(llvm::ArrayRef<host_parsing::BarrierWrapper> barrierConfigs);
    void setConstData(llvm::ArrayRef<uint8_t> weights);

    std::vector<char> getBlob();

private:
    void setNetworkIO(llvm::ArrayRef<mlir::ShapedType> inputsOrOutputs, uint8_t symbolType,
                      elf::writer::SymbolSection*& symbolSection, const std::string& symbolName);

    void finalize();
    void finalizeDMA();
    void finalizeDPU();

private:
    class RelocationManager {
    public:
        RelocationManager(elf::writer::Section* sectionToPatch, const std::string& relocationSectionName,
                          ELFBlobSerializer& elfBlobSerializer);

        void addRelocation(const TensorPatchingInfo& tensorPatchingInfo, elf::Elf_Word type,
                           elf::Elf64_Addr sectionOffset, OffsetToUse offsetToUse = OffsetToUse::DATA);
        void addRelocation(const elf::writer::SymbolSection* symbolSection, const elf::writer::Symbol* symbol,
                           elf::Elf_Word type, elf::Elf_Sxword addend, elf::Elf64_Addr sectionOffset);
        void addRelocation(elf::Elf_Word specialSymbol, elf::Elf_Word type, elf::Elf_Sxword addend,
                           elf::Elf64_Addr sectionOffset);

    private:
        elf::writer::Relocation* addRelocation(elf::writer::RelocationSection* relocationSection,
                                               const elf::writer::Symbol* symbol, elf::Elf_Word type,
                                               elf::Elf_Sxword addend, elf::Elf64_Addr offset);
        elf::writer::RelocationSection* getRelocationSection(const elf::writer::SymbolSection* symbolSection);
        elf::writer::RelocationSection* getRelocationSection(const elf::writer::SymbolSection* symbolSection,
                                                             VPUIP::MemoryLocation memoryLocation);
        elf::writer::RelocationSection* createRelocationSection(const elf::writer::SymbolSection* symbolSection);

    private:
        elf::writer::Section* m_sectionToPatch = nullptr;
        std::string m_relocationSectionName;
        ELFBlobSerializer& m_elfBlobSerializer;

        elf::writer::RelocationSection* m_specialSymbolRelocation = nullptr;
        std::unordered_map<const elf::writer::SymbolSection*, elf::writer::RelocationSection*>
                m_symbolTableToRelocation;
    };

private:
    void updateInvariant(DPUInvariantTask& invariantTask, RelocationManager& relocationManager,
                         uint64_t invariantSectionOffset);
    void updateInvariantSOH(DPUInvariantTask& invariantTask, RelocationManager& relocationManager,
                            uint64_t invariantSectionOffset);

private:
    elf::Writer m_writer;

    elf::writer::EmptySection* m_ddrScratch;
    host_parsing::ResourceRequirements m_resourceRequirements;
    host_parsing::MappedInference m_mappedInference;

    std::array<std::vector<DmaTask>, 2> m_dmaTasks;
    std::vector<DPUTask> m_dpuTasks;

    elf::writer::SymbolSection* m_networkInputSymbols = nullptr;
    elf::writer::SymbolSection* m_networkOutputSymbols = nullptr;
    elf::writer::SymbolSection* m_sectionSymbols = nullptr;
    std::unordered_map<elf::writer::Section*, elf::writer::Symbol*> m_sectionSymbolsMapping;

    std::array<elf::writer::BinaryDataSection<host_parsing::DmaWrapper>*, 2> m_dmaTasksSections = {nullptr};
    elf::writer::BinaryDataSection<host_parsing::DPUInvariantWrapper>* m_dpuInvariants = nullptr;
    elf::writer::BinaryDataSection<host_parsing::DPUVariantWrapper>* m_dpuVariants = nullptr;
    elf::writer::BinaryDataSection<host_parsing::BarrierWrapper>* m_barrierConfigs = nullptr;
    elf::writer::BinaryDataSection<uint8_t>* m_weights = nullptr;

    // ALL ABOUT KERNELS
    elf::Reader32 m_reader;
    bool isReaderInit = false;

    size_t m_inputElfSecNum = 0;

    int m_kernelsNum = 0;
    std::string m_kernelName = "actKernel";
    elf::writer::SymbolSection* m_actKernel_symbols = nullptr;
    std::map<std::string, std::vector<std::string>> m_actKernelsMapping;

    // Act Kernel Ranges
    elf::writer::Symbol* m_actKernelRangeSymbol = nullptr;
    elf::writer::BinaryDataSection<host_parsing::ActKernelRange>* m_temp_actKernelRanges = nullptr;
    elf::writer::RelocationSection* m_temp_actKernelRangeRela = nullptr;
    elf::writer::BinaryDataSection<host_parsing::ActKernelRangeWrapper>* m_actKernelRanges = nullptr;
    elf::writer::RelocationSection* m_actKernelRangeRela = nullptr;

    // Act Kernel Invocations
    elf::writer::Symbol* m_actKernelInvocationSymbol = nullptr;
    elf::writer::BinaryDataSection<host_parsing::ActKernelInvocation>* m_temp_actKernelInvocations = nullptr;
    elf::writer::RelocationSection* m_temp_actKernelInvocationRela = nullptr;
    elf::writer::BinaryDataSection<host_parsing::ActKernelInvocationWrapper>* m_actKernelInvocations = nullptr;
    elf::writer::RelocationSection* m_actKernelInvocationRela = nullptr;
};

}  // namespace VPUIP
}  // namespace vpux
