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
    m_symbols = m_writer.addSymbolSection();
    m_symbols->setName(".symtab");

    m_mappedInference.barrierConfigs.count = 0;
    m_mappedInference.dmaTasks[0].count = 0;
    m_mappedInference.dmaTasks[1].count = 0;
    m_mappedInference.leadingDmaCount[0] = 0;
    m_mappedInference.leadingDmaCount[1] = 0;
    m_mappedInference.variants.count = 0;
    m_mappedInference.invariants.count = 0;
}

void VPUIP::ELFBlobSerializer::setDDRScratch(size_t ddrScratch) {
    m_ddrScratch = ddrScratch;
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

void VPUIP::ELFBlobSerializer::setLeadingDMACount0(uint32_t leadingDMACount) {
    m_mappedInference.leadingDmaCount[0] = leadingDMACount;
}

void VPUIP::ELFBlobSerializer::setDMATasks0(llvm::ArrayRef<std::pair<DmaWrapper, DMATaskExtension>> dmaTasks) {
    m_mappedInference.dmaTasks[0].count = dmaTasks.size();

    std::vector<DmaWrapper> dmaDescriptors;
    dmaDescriptors.reserve(dmaTasks.size());
    for (const auto& dmaTask : dmaTasks) {
        dmaDescriptors.push_back(dmaTask.first);
    }

    m_dmaTasks0 = m_writer.addBinaryDataSection<DmaWrapper>();
    m_dmaTasks0->addData(dmaDescriptors.data(), dmaDescriptors.size());
    m_dmaTasks0->setName(".text.DMATasks0");
    m_dmaTasks0->setFlags(SHF_EXECINSTR);
    m_dmaTasks0->setAddrAlign(64);

    std::map<elf::writer::SymbolSection*, elf::writer::RelocationSection*> symbolTableToRelocation;

    const auto addTensorPtrRelocation = [&symbolTableToRelocation, this](Elf_Word type,
                                                                         const TensorPatchingInfo& tensorPatchingInfo,
                                                                         Elf64_Addr offset) {
        const auto symTab = getSymbolSection(tensorPatchingInfo.location);
        if (symbolTableToRelocation.find(symTab) == symbolTableToRelocation.end()) {
            auto relocationSection = createRelocationSection(tensorPatchingInfo.location.memLocation);
            relocationSection->setName(".rela.DMA");
            relocationSection->setSectionToPatch(m_dmaTasks0);
            relocationSection->setSymbolTable(symTab);
            symbolTableToRelocation[symTab] = relocationSection;
        }

        auto relocationSection = symbolTableToRelocation.at(symTab);
        auto relocation = relocationSection->addRelocationEntry();
        relocation->setType(type);
        relocation->setSymbol(getSymbol(tensorPatchingInfo.location));
        relocation->setAddend(tensorPatchingInfo.offset);
        relocation->setOffset(offset);
    };

    for (size_t i = 0; i < dmaTasks.size(); i++) {
        const auto& extension = dmaTasks[i].second;

        addTensorPtrRelocation(
                R_VPU_64, extension.input,
                i * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction) + offsetof(DmaDescriptor, src));
        addTensorPtrRelocation(
                R_VPU_64, extension.output,
                i * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction) + offsetof(DmaDescriptor, dst));

        //        if (extension.linkAddress.location.type != LocationType::NONE) {
        //            addRelocation(extension.linkAddress.location.type == LocationType::DDR_DMA ? R_VPU_64_OR :
        //            R_VPU_64_OR_RTM,
        //                          getSymbol(extension.linkAddress.location),
        //                          sizeof(DmaWrapper) * extension.linkAddress.offset + offsetof(DmaWrapper,
        //                          transaction), i * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction));
    }
}

void VPUIP::ELFBlobSerializer::setBarrierConfigs(llvm::ArrayRef<BarrierWrapper> barrierConfigs) {
    m_mappedInference.barrierConfigs.count = barrierConfigs.size();

    m_barrierConfigs = m_writer.addBinaryDataSection<BarrierWrapper>();
    m_barrierConfigs->addData(barrierConfigs.data(), barrierConfigs.size());
    m_barrierConfigs->setName(".text.BarrierConfigs");
    m_barrierConfigs->setFlags(SHF_EXECINSTR);
    m_barrierConfigs->setAddrAlign(64);
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
                                            elf::writer::SymbolSection*& symbolSection, const std::string& symbolName) {
    symbolSection = m_writer.addSymbolSection();
    symbolSection->setName(symbolName + "s");

    for (size_t i = 0; i < inputsOrOutputs.size(); i++) {
        const auto& inputOrOutput = inputsOrOutputs[i];

        auto inputOrOutputSym = symbolSection->addSymbolEntry();
        inputOrOutputSym->setName(symbolName);  // TODO: get name of tensor?
        inputOrOutputSym->setType(symbolType);  // TODO:: do we need to reserve a type for I/O ?
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

    if (m_ddrScratch) {
        auto ddrScratch = m_writer.addEmptySection();
        ddrScratch->setName(".ddr.Scratch");
        ddrScratch->setType(VPU_SHT_DDR);
        ddrScratch->setFlags(SHF_ALLOC);
        ddrScratch->setSize(m_ddrScratch);
    }

    auto tilesCount = m_writer.addEmptySection();
    tilesCount->setName(".tiles");
    tilesCount->setType(VPU_SHT_TILES);
    tilesCount->setFlags(SHF_ALLOC);
    tilesCount->setSize(m_resourceRequirements.slice_count);

    //
    // MappedInference
    //

    auto mappedInferenceSection = m_writer.addBinaryDataSection<MappedInference>();
    mappedInferenceSection->setName(".text.MappedInference");
    mappedInferenceSection->addData(m_mappedInference);
    mappedInferenceSection->setFlags(SHF_EXECINSTR);
    mappedInferenceSection->setAddrAlign(64);

    auto mappedInferenceSymbol = m_symbols->addSymbolEntry();
    mappedInferenceSymbol->setName("mappedInference");
    mappedInferenceSymbol->setRelatedSection(mappedInferenceSection);
    mappedInferenceSymbol->setType(STT_SECTION);
    mappedInferenceSymbol->setSize(mappedInferenceSection->getDataSize());

    auto mappedInferenceRelaSection = m_writer.addRelocationSection();
    mappedInferenceRelaSection->setName(".rela.MappedInference");
    mappedInferenceRelaSection->setSymbolTable(m_symbols);
    mappedInferenceRelaSection->setSectionToPatch(mappedInferenceSection);

    auto entryPoint = m_symbols->addSymbolEntry();
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

    if (m_dmaTasks0) {
        auto dmaTasksSymbol = m_symbols->addSymbolEntry();
        dmaTasksSymbol->setName(".ddr.dmaTasks0");
        dmaTasksSymbol->setRelatedSection(m_dmaTasks0);
        dmaTasksSymbol->setType(STT_SECTION);
        dmaTasksSymbol->setSize(m_dmaTasks0->getDataSize());

        auto mappedInferenceDMARelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceDMARelocation->setSymbol(dmaTasksSymbol);
        mappedInferenceDMARelocation->setType(R_VPU_64);
        mappedInferenceDMARelocation->setOffset(offsetof(MappedInference, dmaTasks[0]) +
                                                offsetof(TaskReference<DmaWrapper>, address));
        mappedInferenceDMARelocation->setAddend(0);
        segment->addSection(m_dmaTasks0);
    }

    //
    // BarrierConfigs
    //

    if (m_barrierConfigs) {
        auto barrierConfigsSymbol = m_symbols->addSymbolEntry();
        barrierConfigsSymbol->setName(".ddr.barrierConfigs");
        barrierConfigsSymbol->setRelatedSection(m_dmaTasks0);
        barrierConfigsSymbol->setType(STT_SECTION);
        barrierConfigsSymbol->setSize(m_dmaTasks0->getDataSize());

        auto mappedInferenceBarrierConfigsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceBarrierConfigsRelocation->setSymbol(barrierConfigsSymbol);
        mappedInferenceBarrierConfigsRelocation->setType(R_VPU_64);
        mappedInferenceBarrierConfigsRelocation->setOffset(offsetof(MappedInference, barrierConfigs) +
                                                           offsetof(TaskReference<DmaWrapper>, address));
        mappedInferenceBarrierConfigsRelocation->setAddend(0);
        segment->addSection(m_barrierConfigs);
    }
}

elf::writer::SymbolSection* VPUIP::ELFBlobSerializer::getSymbolSection(const TensorLocation& location) {
    switch (location.memLocation) {
    case VPUIP::MemoryLocation::ProgrammableInput:
        return m_networkInputSymbols;
    case VPUIP::MemoryLocation::ProgrammableOutput:
        return m_networkOutputSymbols;
    default: {
        if (m_symbols == nullptr) {
            m_symbols = m_writer.addSymbolSection();
            m_symbols->setName(".symtab");
        }
        return m_symbols;
    }
    }
}

elf::writer::Symbol* VPUIP::ELFBlobSerializer::getSymbol(VPUIP::MemoryLocation location) {
    if (m_symbolsMapping.find(location) != m_symbolsMapping.end()) {
        return m_symbolsMapping.at(location);
    }

    auto symbol = m_symbols->addSymbolEntry();
    symbol->setName(stringifyMemoryLocation(location).str());
    return symbol;
}

writer::Symbol* VPUIP::ELFBlobSerializer::getSymbol(const TensorLocation& location) {
    switch (location.memLocation) {
    case VPUIP::MemoryLocation::ProgrammableInput:
    case VPUIP::MemoryLocation::ProgrammableOutput:
        return getSymbolSection(location)->getSymbols()[location.locationIndex + 1].get();
    default:
        return getSymbol(location.memLocation);
    }
}

elf::writer::RelocationSection* VPUIP::ELFBlobSerializer::createRelocationSection(VPUIP::MemoryLocation location) {
    auto relocation = m_writer.addRelocationSection();
    if (location == VPUIP::MemoryLocation::ProgrammableInput || location == VPUIP::MemoryLocation::ProgrammableOutput) {
        relocation->setFlags(VPU_SHF_JIT);
    }

    return relocation;
}
