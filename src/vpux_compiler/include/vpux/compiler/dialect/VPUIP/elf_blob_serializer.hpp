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

#include <vpux/compiler/dialect/VPUIP/attributes/enums.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <host_parsed_inference.h>

#include <unordered_map>

namespace vpux {
namespace VPUIP {

struct TensorLocation {
    VPUIP::MemoryLocation memLocation;
    uint64_t locationIndex = 0;
};

struct TensorPatchingInfo {
    TensorLocation location;
    uint64_t offset = 0;
};

struct LinkAddressPatchingInfo {
    enum class MetaDataLocation { NONE = 0, DDR_DMA, RTM_DMA } metaDataLocation = MetaDataLocation::NONE;
    size_t dmaTaskIndex = 0;
};

struct DmaTask {
    DmaWrapper dmaDescriptor{};
    TensorPatchingInfo input{};
    TensorPatchingInfo output{};
    LinkAddressPatchingInfo linkAddress{};
};

class ELFBlobSerializer {
public:
    ELFBlobSerializer();

    void setNetworkInputs(llvm::ArrayRef<mlir::MemRefType> inputs);
    void setNetworkOutputs(llvm::ArrayRef<mlir::MemRefType> outputs);

    void setDDRScratch(size_t ddrScratch);
    void setResourceRequirements(const ResourceRequirements& resourceRequirements);

    void setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex = 0);
    void setDMATasks(llvm::ArrayRef<DmaTask> dmaTasks, size_t dmaEngineIndex = 0);
    void setBarrierConfigs(llvm::ArrayRef<BarrierWrapper> barrierConfigs);

    std::vector<char> getBlob();
    void write(const std::string& fileName);

private:
    void setNetworkIO(llvm::ArrayRef<mlir::MemRefType> inputsOrOutputs, uint8_t symbolType,
                      elf::writer::SymbolSection*& symbolSection, const std::string& symbolName);
    void finalize();

private:
    class RelocationManager {
    public:
        using SymbolInfo = struct {
            elf::writer::SymbolSection* symbolSection;
            elf::writer::Symbol* symbol;
        };

    public:
        RelocationManager(elf::writer::Section* sectionToPatch, const std::string& relocationSectionName,
                          ELFBlobSerializer& elfBlobSerializer);

        void addRelocation(const TensorPatchingInfo& tensorPatchingInfo, elf::Elf_Word type,
                           elf::Elf64_Addr sectionOffset);
        void addRelocation(const elf::writer::SymbolSection* symbolSection, const elf::writer::Symbol* symbol,
                           elf::Elf_Word type, elf::Elf_Sxword addend, elf::Elf64_Addr sectionOffset);
        void addRelocation(elf::Elf_Word specialSymbol, elf::Elf_Word type, elf::Elf_Sxword addend,
                           elf::Elf64_Addr sectionOffset);

    private:
        elf::writer::Relocation* addRelocation(elf::writer::RelocationSection* relocationSection,
                                               const elf::writer::Symbol* symbol, elf::Elf_Word type,
                                               elf::Elf_Sxword addend, elf::Elf64_Addr offset);
        elf::writer::RelocationSection* createRelocationSection(const elf::writer::SymbolSection* symbolSection);
        SymbolInfo getSymbolInfo(const TensorLocation& location);

    private:
        elf::writer::Section* m_sectionToPatch = nullptr;
        std::string m_relocationSectionName;
        ELFBlobSerializer& m_elfBlobSerializer;

        elf::writer::RelocationSection* m_specialSymbolRelocation = nullptr;
        std::unordered_map<const elf::writer::SymbolSection*, elf::writer::RelocationSection*>
                m_symbolTableToRelocation;
    };

private:
    elf::Writer m_writer;

    elf::writer::EmptySection* m_ddrScratch;
    ResourceRequirements m_resourceRequirements;
    MappedInference m_mappedInference;

    elf::writer::SymbolSection* m_networkInputSymbols = nullptr;
    elf::writer::SymbolSection* m_networkOutputSymbols = nullptr;
    elf::writer::SymbolSection* m_sectionSymbols = nullptr;
    std::unordered_map<elf::writer::Section*, elf::writer::Symbol*> m_sectionSymbolsMapping;

    std::array<elf::writer::BinaryDataSection<DmaWrapper>*, 2> m_dmaTasks = {nullptr};
    elf::writer::BinaryDataSection<BarrierWrapper>* m_barrierConfigs = nullptr;
};

}  // namespace VPUIP
}  // namespace vpux
