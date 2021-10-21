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

#include <vpux/compiler/dialect/VPUIP/elf_blob_serializer.hpp>

#include <vpux/compiler/utils/types.hpp>

#include <elf/types/vpu_extensions.hpp>

using namespace vpux;
using namespace elf;

VPUIP::ELFBlobSerializer::ELFBlobSerializer() {
    m_sectionSymbols = m_writer.addSymbolSection();
    m_sectionSymbols->setName(".symtab");

    m_mappedInference.barrierConfigs.count = 0;
    m_mappedInference.dmaTasks[0].count = 0;
    m_mappedInference.dmaTasks[1].count = 0;
    m_mappedInference.leadingDmaCount[0] = 0;
    m_mappedInference.leadingDmaCount[1] = 0;
    m_mappedInference.variants.count = 0;
    m_mappedInference.invariants.count = 0;
}

void VPUIP::ELFBlobSerializer::setDDRScratch(size_t ddrScratch) {
    m_ddrScratch = m_writer.addEmptySection();
    m_ddrScratch->setName(".ddr.Scratch");
    m_ddrScratch->setType(VPU_SHT_DDR);
    m_ddrScratch->setFlags(SHF_ALLOC);
    m_ddrScratch->setSize(ddrScratch);

    auto ddrScratchSymbol = m_sectionSymbols->addSymbolEntry();
    ddrScratchSymbol->setName(".ddr.Scratch");
    ddrScratchSymbol->setRelatedSection(m_ddrScratch);
    ddrScratchSymbol->setType(STT_SECTION);
    ddrScratchSymbol->setSize(m_ddrScratch->getDataSize());
    m_sectionSymbolsMapping.insert(std::make_pair(m_ddrScratch, ddrScratchSymbol));
}

void VPUIP::ELFBlobSerializer::setResourceRequirements(const ResourceRequirements& resourceRequirements) {
    m_resourceRequirements = resourceRequirements;
}

void VPUIP::ELFBlobSerializer::setNetworkInputs(llvm::ArrayRef<mlir::MemRefType> inputs) {
    setNetworkIO(inputs, VPU_STT_INPUT, m_networkInputSymbols, "input");
}

void VPUIP::ELFBlobSerializer::setNetworkOutputs(llvm::ArrayRef<mlir::MemRefType> outputs) {
    setNetworkIO(outputs, VPU_STT_OUTPUT, m_networkOutputSymbols, "output");
}

void VPUIP::ELFBlobSerializer::setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex) {
    m_mappedInference.leadingDmaCount[dmaEngineIndex] = leadingDMACount;
}

void VPUIP::ELFBlobSerializer::setDMATasks(llvm::ArrayRef<DmaTask> dmaTasks, size_t dmaEngineIndex) {
    m_mappedInference.dmaTasks[dmaEngineIndex].count = dmaTasks.size();

    std::vector<DmaWrapper> dmaDescriptors;
    dmaDescriptors.reserve(dmaTasks.size());
    for (const auto& dmaTask : dmaTasks) {
        dmaDescriptors.push_back(dmaTask.dmaDescriptor);
    }

    auto dmaTasksSection = m_dmaTasks[dmaEngineIndex];
    dmaTasksSection = m_writer.addBinaryDataSection<DmaWrapper>();
    dmaTasksSection->addData(dmaDescriptors.data(), dmaDescriptors.size());
    dmaTasksSection->setName(".text.DMATasks" + std::to_string(dmaEngineIndex));
    dmaTasksSection->setFlags(SHF_EXECINSTR);
    dmaTasksSection->setAddrAlign(64);

    auto dmaTasksSymbol = m_sectionSymbols->addSymbolEntry();
    dmaTasksSymbol->setName(".ddr.dmaTasks" + std::to_string(dmaEngineIndex));
    dmaTasksSymbol->setRelatedSection(dmaTasksSection);
    dmaTasksSymbol->setType(STT_SECTION);
    dmaTasksSymbol->setSize(dmaTasksSection->getDataSize());
    m_sectionSymbolsMapping.insert(std::make_pair(dmaTasksSection, dmaTasksSymbol));

    RelocationManager relocationManager(dmaTasksSection, ".rela.DMA", *this);

    for (size_t i = 0; i < dmaTasks.size(); ++i) {
        const auto& dmaTask = dmaTasks[i];
        auto& dmaDescriptor = dmaDescriptors[i];

        const auto transactionOffset = i * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction);
        relocationManager.addRelocation(dmaTask.input, R_VPU_64, transactionOffset + offsetof(DmaDescriptor, src));
        relocationManager.addRelocation(dmaTask.output, R_VPU_64, transactionOffset + offsetof(DmaDescriptor, dst));

        if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::DDR_DMA) {
            relocationManager.addRelocation(
                    m_sectionSymbols, dmaTasksSymbol, R_VPU_64_OR,
                    dmaTask.linkAddress.dmaTaskIndex * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction),
                    transactionOffset);
        } else if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::RTM_DMA) {
            dmaDescriptor.transaction.link_address = dmaTask.linkAddress.dmaTaskIndex;
            relocationManager.addRelocation(NNRD_SYM_RTM_DMA0 + dmaEngineIndex, R_VPU_64_OR_RTM, sizeof(DmaWrapper),
                                            transactionOffset);
        }

        if (dmaTask.dmaDescriptor.transaction.barriers.prod_mask) {
            relocationManager.addRelocation(NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                                            +offsetof(DmaDescriptor, barriers) + offsetof(DmaBarrierCfg, prod_mask));
        }

        if (dmaTask.dmaDescriptor.transaction.barriers.cons_mask) {
            relocationManager.addRelocation(NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                                            +offsetof(DmaDescriptor, barriers) + offsetof(DmaBarrierCfg, cons_mask));
        }
    }
}

void VPUIP::ELFBlobSerializer::setBarrierConfigs(llvm::ArrayRef<BarrierWrapper> barrierConfigs) {
    m_mappedInference.barrierConfigs.count = barrierConfigs.size();

    m_barrierConfigs = m_writer.addBinaryDataSection<BarrierWrapper>();
    m_barrierConfigs->addData(barrierConfigs.data(), barrierConfigs.size());
    m_barrierConfigs->setName(".text.BarrierConfigs");
    m_barrierConfigs->setFlags(SHF_EXECINSTR);
    m_barrierConfigs->setAddrAlign(64);

    auto barrierConfigsSymbol = m_sectionSymbols->addSymbolEntry();
    barrierConfigsSymbol->setName(".ddr.barrierConfigs");
    barrierConfigsSymbol->setRelatedSection(m_barrierConfigs);
    barrierConfigsSymbol->setType(STT_SECTION);
    barrierConfigsSymbol->setSize(m_barrierConfigs->getDataSize());
    m_sectionSymbolsMapping.insert(std::make_pair(m_barrierConfigs, barrierConfigsSymbol));
}

std::vector<char> VPUIP::ELFBlobSerializer::getBlob() {
    finalize();
    std::vector<char> res;
    m_writer.write(res);
    return res;
}

void VPUIP::ELFBlobSerializer::write(const std::string& fileName) {
    finalize();
    m_writer.write(fileName);
}

void VPUIP::ELFBlobSerializer::setNetworkIO(llvm::ArrayRef<mlir::MemRefType> inputsOrOutputs, uint8_t symbolType,
                                            writer::SymbolSection*& symbolSection, const std::string& symbolName) {
    symbolSection = m_writer.addSymbolSection();
    symbolSection->setName(symbolName + "s");

    for (size_t i = 0; i < inputsOrOutputs.size(); i++) {
        const auto& inputOrOutput = inputsOrOutputs[i];

        auto inputOrOutputSym = symbolSection->addSymbolEntry();
        inputOrOutputSym->setName(symbolName);  // TODO: get name of tensor?
        inputOrOutputSym->setType(symbolType);
        inputOrOutputSym->setValue(i);
        inputOrOutputSym->setSize(inputOrOutput.getSizeInBits());

        TensorLocation loc;
        loc.memLocation = VPUIP::MemoryLocation::ProgrammableInput;
        loc.locationIndex = i;
    }
}

void VPUIP::ELFBlobSerializer::finalize() {
    //
    // Resources
    //

    auto tilesCount = m_writer.addEmptySection();
    tilesCount->setName(".tiles");
    tilesCount->setType(VPU_SHT_TILES);
    tilesCount->setSize(m_resourceRequirements.slice_count);

    //
    // MappedInference
    //

    auto mappedInferenceSection = m_writer.addBinaryDataSection<MappedInference>();
    mappedInferenceSection->setName(".text.MappedInference");
    mappedInferenceSection->addData(m_mappedInference);
    mappedInferenceSection->setFlags(SHF_EXECINSTR);
    mappedInferenceSection->setAddrAlign(64);

    auto mappedInferenceSymbol = m_sectionSymbols->addSymbolEntry();
    mappedInferenceSymbol->setName(".ddr.mappedInference");
    mappedInferenceSymbol->setRelatedSection(mappedInferenceSection);
    mappedInferenceSymbol->setType(STT_SECTION);
    mappedInferenceSymbol->setSize(mappedInferenceSection->getDataSize());

    auto mappedInferenceRelaSection = m_writer.addRelocationSection();
    mappedInferenceRelaSection->setName(".rela.MappedInference");
    mappedInferenceRelaSection->setSymbolTable(m_sectionSymbols);
    mappedInferenceRelaSection->setSectionToPatch(mappedInferenceSection);

    auto entryPoint = m_sectionSymbols->addSymbolEntry();
    entryPoint->setName(".start");
    entryPoint->setRelatedSection(mappedInferenceSection);
    entryPoint->setType(VPU_STT_ENTRY);

    auto segment = m_writer.addSegment();
    segment->setType(PT_LOAD);
    segment->setAlign(64);
    segment->addSection(mappedInferenceSection);

    //
    // DMATasks
    //

    for (size_t i = 0; i < m_dmaTasks.size(); ++i) {
        const auto& dmaTasks = m_dmaTasks[i];
        if (dmaTasks) {
            auto mappedInferenceDMARelocation = mappedInferenceRelaSection->addRelocationEntry();
            mappedInferenceDMARelocation->setSymbol(m_sectionSymbolsMapping.at(dmaTasks));
            mappedInferenceDMARelocation->setType(R_VPU_64);
            mappedInferenceDMARelocation->setOffset(offsetof(MappedInference, dmaTasks) + sizeof(dmaTasks) * i +
                                                    offsetof(TaskReference<DmaWrapper>, address));
            mappedInferenceDMARelocation->setAddend(0);
            segment->addSection(dmaTasks);
        }
    }

    //
    // BarrierConfigs
    //

    if (m_barrierConfigs) {
        auto mappedInferenceBarrierConfigsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceBarrierConfigsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_barrierConfigs));
        mappedInferenceBarrierConfigsRelocation->setType(R_VPU_64);
        mappedInferenceBarrierConfigsRelocation->setOffset(offsetof(MappedInference, barrierConfigs) +
                                                           offsetof(TaskReference<DmaWrapper>, address));
        mappedInferenceBarrierConfigsRelocation->setAddend(0);
        segment->addSection(m_barrierConfigs);
    }
}

VPUIP::ELFBlobSerializer::RelocationManager::RelocationManager(writer::Section* sectionToPatch,
                                                               const std::string& relocationSectionName,
                                                               ELFBlobSerializer& elfBlobSerializer)
        : m_sectionToPatch(sectionToPatch),
          m_relocationSectionName(relocationSectionName),
          m_elfBlobSerializer(elfBlobSerializer) {
}

void VPUIP::ELFBlobSerializer::RelocationManager::addRelocation(const TensorPatchingInfo& tensorPatchingInfo,
                                                                Elf_Word type, Elf64_Addr sectionOffset) {
    const auto symbolInfo = getSymbolInfo(tensorPatchingInfo.location);
    if (m_symbolTableToRelocation.find(symbolInfo.symbolSection) == m_symbolTableToRelocation.end()) {
        auto relocationTable = createRelocationSection(symbolInfo.symbolSection);
        if (tensorPatchingInfo.location.memLocation == VPUIP::MemoryLocation::ProgrammableInput ||
            tensorPatchingInfo.location.memLocation == VPUIP::MemoryLocation::ProgrammableOutput) {
            relocationTable->setFlags(VPU_SHF_JIT);
        }
        m_symbolTableToRelocation[symbolInfo.symbolSection] = relocationTable;
    }

    addRelocation(m_symbolTableToRelocation.at(symbolInfo.symbolSection), symbolInfo.symbol, type,
                  tensorPatchingInfo.offset, sectionOffset);
}

void VPUIP::ELFBlobSerializer::RelocationManager::addRelocation(const writer::SymbolSection* symbolSection,
                                                                const writer::Symbol* symbol, Elf_Word type,
                                                                Elf_Sxword addend, Elf64_Addr sectionOffset) {
    addRelocation(m_symbolTableToRelocation.at(symbolSection), symbol, type, addend, sectionOffset);
}

void VPUIP::ELFBlobSerializer::RelocationManager::addRelocation(Elf_Word specialSymbol, Elf_Word type,
                                                                Elf_Sxword addend, Elf64_Addr sectionOffset) {
    if (m_specialSymbolRelocation == nullptr) {
        m_specialSymbolRelocation = createRelocationSection(nullptr);
        m_specialSymbolRelocation->setSpecialSymbolTable(VPU_RT_SYMTAB);
    }

    auto relocation = addRelocation(m_specialSymbolRelocation, nullptr, type, addend, sectionOffset);
    relocation->setSpecialSymbol(specialSymbol);
}

writer::Relocation* VPUIP::ELFBlobSerializer::RelocationManager::addRelocation(
        writer::RelocationSection* relocationSection, const writer::Symbol* symbol, Elf_Word type, Elf_Sxword addend,
        Elf64_Addr sectionOffset) {
    auto relocation = relocationSection->addRelocationEntry();
    relocation->setType(type);
    relocation->setSymbol(symbol);
    relocation->setAddend(addend);
    relocation->setOffset(sectionOffset);
    return relocation;
}

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::createRelocationSection(
        const writer::SymbolSection* symbolSection) {
    auto relocationTable = m_elfBlobSerializer.m_writer.addRelocationSection();
    relocationTable->setName(m_relocationSectionName);
    relocationTable->setSectionToPatch(m_sectionToPatch);
    relocationTable->setSymbolTable(symbolSection);
    return relocationTable;
}

VPUIP::ELFBlobSerializer::RelocationManager::SymbolInfo VPUIP::ELFBlobSerializer::RelocationManager::getSymbolInfo(
        const TensorLocation& location) {
    SymbolInfo symbolInfo;

    switch (location.memLocation) {
    case VPUIP::MemoryLocation::ProgrammableInput:
        symbolInfo.symbolSection = m_elfBlobSerializer.m_networkInputSymbols;
        symbolInfo.symbol = m_elfBlobSerializer.m_networkInputSymbols->getSymbols()[location.locationIndex + 1].get();
        break;
    case VPUIP::MemoryLocation::ProgrammableOutput:
        symbolInfo.symbolSection = m_elfBlobSerializer.m_networkOutputSymbols;
        symbolInfo.symbol = m_elfBlobSerializer.m_networkOutputSymbols->getSymbols()[location.locationIndex + 1].get();
        break;
    case VPUIP::MemoryLocation::VPU_DDR_BSS:
        symbolInfo.symbolSection = m_elfBlobSerializer.m_sectionSymbols;
        symbolInfo.symbol = m_elfBlobSerializer.m_sectionSymbolsMapping.at(m_elfBlobSerializer.m_ddrScratch);
        break;
    default:
        VPUX_THROW("Unsupported MemoryLocation {}", location.memLocation);
        break;
    }

    return symbolInfo;
}
