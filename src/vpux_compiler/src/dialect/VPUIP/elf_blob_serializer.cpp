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

    m_mappedInference.actKRanges.count = 0;
    m_mappedInference.actKInvocations.count = 0;
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

//************ ACT KERNELS ************

void VPUIP::ELFBlobSerializer::initActKernel(std::vector<char> elfBlob, std::string name){

    m_reader.loadElf(elfBlob.data(), elfBlob.size());

    isReaderInit = true;

    m_kernelName = name;

    m_kernelsNum++;

    m_actKernel_symbols = m_writer.addSymbolSection();
    m_actKernel_symbols->setName(".symtab.ActKernel");


    // Act Kernel Ranges init
    m_temp_actKernelRanges = m_writer.addBinaryDataSection<ActKernelRange>();
    m_temp_actKernelRanges->setName(".text." + m_kernelName + ".ActKernelRange");

    m_actKernelRangeSymbol = m_actKernel_symbols->addSymbolEntry();
    m_actKernelRangeSymbol->setName(m_kernelName + ".actKernelRange");
    m_actKernelRangeSymbol->setRelatedSection(m_temp_actKernelRanges);
    m_actKernelRangeSymbol->setType(STT_SECTION);
    m_actKernelRangeSymbol->setSize(sizeof(ActKernelRange));

    m_temp_actKernelRangeRela = m_writer.addRelocationSection();
    m_temp_actKernelRangeRela->setName(".rela." + m_kernelName + ".ActKernelRange");
    m_temp_actKernelRangeRela->setSymbolTable(m_actKernel_symbols);
    m_temp_actKernelRangeRela->setSectionToPatch(m_temp_actKernelRanges);

    m_actKernelRanges = m_writer.addBinaryDataSection<ActKernelRangeWrapper>();
    m_actKernelRanges->setName(".text." + m_kernelName + ".ActKernelRangeWrapper");

    m_actKernelRangeRela = m_writer.addRelocationSection();
    m_actKernelRangeRela->setName(".rela." + m_kernelName + ".ActKernelRangeWrapper");
    m_actKernelRangeRela->setSymbolTable(m_actKernel_symbols);
    m_actKernelRangeRela->setSectionToPatch(m_actKernelRanges);


    // Act Kernel Invocations init
    m_temp_actKernelInvocations = m_writer.addBinaryDataSection<ActKernelInvocation>();
    m_temp_actKernelInvocations->setName(".text." + m_kernelName + ".ActKernelInvocation");

    m_actKernelInvocationSymbol = m_actKernel_symbols->addSymbolEntry();
    m_actKernelInvocationSymbol->setName(m_kernelName + ".actKernelInvocation");
    m_actKernelInvocationSymbol->setRelatedSection(m_temp_actKernelInvocations);
    m_actKernelInvocationSymbol->setType(STT_SECTION);
    m_actKernelInvocationSymbol->setSize(sizeof(ActKernelInvocation));

    m_temp_actKernelInvocationRela = m_writer.addRelocationSection();
    m_temp_actKernelInvocationRela->setName(".rela." + m_kernelName + ".ActKernelInvocation");
    m_temp_actKernelInvocationRela->setSymbolTable(m_actKernel_symbols);
    m_temp_actKernelInvocationRela->setSectionToPatch(m_temp_actKernelInvocations);

    m_actKernelInvocations = m_writer.addBinaryDataSection<ActKernelInvocationWrapper>();
    m_actKernelInvocations->setName(".text." + m_kernelName + ".ActKernelInvocationWrapper");

    m_actKernelInvocationRela = m_writer.addRelocationSection();
    m_actKernelInvocationRela->setName(".rela." + m_kernelName + ".ActKernelInvocationWrapper");
    m_actKernelInvocationRela->setSymbolTable(m_actKernel_symbols);
    m_actKernelInvocationRela->setSectionToPatch(m_actKernelInvocations);
}


void VPUIP::ELFBlobSerializer::addActKernel(){

    if (!isReaderInit){
        std::cout<<"READER NOT INIT\n";
        return;
    }

    ActKernelRange temp_actKernelRange;
    elf::writer::Symbol* actKernelSymbol = nullptr;

    const ELF32Header* actKernelHeader = m_reader.getHeader();
    auto my_actKernelEntry = actKernelHeader->e_entry;

    //set kernelEntry_
    std::memcpy(&(temp_actKernelRange.kernelEntry_), &my_actKernelEntry, sizeof(Elf32_Addr));

    for (size_t i = 0; i < m_reader.getSectionsNum(); ++i) {
        auto section = m_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();


        if (strcmp(sec_name, ".text") == 0){
            temp_actKernelRange.codeSize_ = sectionHeader->sh_size;

            const auto sec_size =  sectionHeader->sh_size;
            const char* sec_data = section.getData<char>();
            
            auto actKernelSection = m_writer.addBinaryDataSection<char>();
            actKernelSection->setName(".text." + m_kernelName);
            actKernelSection->addData(sec_data, sec_size);
            actKernelSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
            actKernelSection->setAddrAlign(16);

            actKernelSymbol = m_actKernel_symbols->addSymbolEntry();
            actKernelSymbol->setName(m_kernelName);
            actKernelSymbol->setRelatedSection(actKernelSection);
            actKernelSymbol->setType(STT_SECTION);
            actKernelSymbol->setSize(actKernelSection->getDataSize());

            std::vector<std::string> empty_vec;
            m_actKernelsMapping[actKernelSection->getName()] = empty_vec;
            
        }

        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0) //just data perhaps
            temp_actKernelRange.dataSecSize_ = sectionHeader->sh_size;
    }


    // ACTUAL STRUCT SERIALIZATION & RELOCS
    m_temp_actKernelRanges->addData(&temp_actKernelRange, sizeof(temp_actKernelRange));
    m_temp_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_temp_actKernelRanges->setAddrAlign(64);

    auto actKernelRangeRelocation = m_temp_actKernelRangeRela->addRelocationEntry();
    actKernelRangeRelocation->setSymbol(actKernelSymbol);
    actKernelRangeRelocation->setType(R_VPU_64);
    actKernelRangeRelocation->setOffset((m_kernelsNum -1) * sizeof(ActKernelRange) + offsetof(ActKernelRange, textWindowBase_));
    actKernelRangeRelocation->setAddend(0);

}

void VPUIP::ELFBlobSerializer::addActInvocation(){

    if (!isReaderInit){
        std::cout<<"READER NOT INIT\n";
        return;
    }

    std::string textSecName = ".text." + m_kernelName;
    
    if (m_actKernelsMapping.find(textSecName) == m_actKernelsMapping.end()){
        addActKernel();
    }

    ActKernelInvocation temp_actKernelInvo;
    elf::writer::Symbol* actKernelDataSymbol = nullptr;

    auto dataSections = m_actKernelsMapping[textSecName];
    int inv_count = dataSections.size();

    std::string dataSecName = ".data." + m_kernelName + std::to_string(inv_count);
    std::string dataSymbolName = m_kernelName + std::to_string(inv_count);

    for (size_t i = 0; i < m_reader.getSectionsNum(); ++i) {
        auto section = m_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();
    
        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps
            const auto sec_size =  sectionHeader->sh_size;
            const char* sec_data = section.getData<char>();
            
            auto actKernelDataSection = m_writer.addBinaryDataSection<char>();
            actKernelDataSection->setName(dataSecName);
            actKernelDataSection->addData(sec_data, sec_size);
            actKernelDataSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
            actKernelDataSection->setAddrAlign(4);

            actKernelDataSymbol = m_actKernel_symbols->addSymbolEntry();
            actKernelDataSymbol->setName(dataSymbolName);
            actKernelDataSymbol->setRelatedSection(actKernelDataSection);
            actKernelDataSymbol->setType(STT_SECTION);
            actKernelDataSymbol->setSize(actKernelDataSection->getDataSize());

            dataSections.push_back(dataSecName);
            m_actKernelsMapping[textSecName] = dataSections;
        } 
        
    }

    // ACTUAL STRUCT SERIALIZATION & RELOCS
    // CREATE 1 SECTION, and then add data every time 
    m_temp_actKernelInvocations->addData(&temp_actKernelInvo, sizeof(temp_actKernelInvo));
    m_temp_actKernelInvocations->setFlags(SHF_EXECINSTR);
    m_temp_actKernelInvocations->setAddrAlign(64);

    auto actKernelInvoRelocation_range = m_temp_actKernelInvocationRela->addRelocationEntry();
    actKernelInvoRelocation_range->setSymbol(m_actKernelRangeSymbol);
    actKernelInvoRelocation_range->setType(R_VPU_64);
    actKernelInvoRelocation_range->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(ActKernelInvocation) + offsetof(ActKernelInvocation, range_));
    actKernelInvoRelocation_range->setAddend(0);

    auto actKernelInvoRelocation_window = m_temp_actKernelInvocationRela->addRelocationEntry();
    actKernelInvoRelocation_window->setSymbol(actKernelDataSymbol);
    actKernelInvoRelocation_window->setType(R_VPU_64);
    actKernelInvoRelocation_window->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(ActKernelInvocation) + offsetof(ActKernelInvocation, dataWindowBase_));
    actKernelInvoRelocation_window->setAddend(0);

}

void VPUIP::ELFBlobSerializer::finalizeActKernelWrappers(){

    std::string textSecName = ".text." + m_kernelName;

    if (m_actKernelsMapping.find(textSecName) == m_actKernelsMapping.end() || m_actKernelsMapping[textSecName].size() < 1){
        std::cout<<"Error. Range or Invocations not yet set up!\n";
        return;
    }


    // RANGE
    ActKernelRangeWrapper final_actKernelRange;

    final_actKernelRange.kInvoCount_ = m_actKernelsMapping[textSecName].size();

    m_actKernelRanges->addData(&final_actKernelRange, sizeof(final_actKernelRange));
    m_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_actKernelRanges->setAddrAlign(64);

    auto actKernelRangeWrapperRelocation = m_actKernelRangeRela->addRelocationEntry();
    actKernelRangeWrapperRelocation->setSymbol(m_actKernelRangeSymbol);
    actKernelRangeWrapperRelocation->setType(R_VPU_64);
    actKernelRangeWrapperRelocation->setOffset((m_kernelsNum -1) * sizeof(ActKernelRangeWrapper) + offsetof(ActKernelRangeWrapper, kRange_));
    actKernelRangeWrapperRelocation->setAddend((m_kernelsNum -1) * sizeof(ActKernelRange));



    // INVOCATION
    int inv_count = m_actKernelsMapping[textSecName].size();

    ActKernelInvocationWrapper final_actKernelInvo;

    final_actKernelInvo.kRangeIndex_ = m_kernelsNum - 1; // set last fields of ActKernelInvoWrapper

    for (int i = 0; i < inv_count; i++){
        m_actKernelInvocations->addData(&final_actKernelInvo, sizeof(final_actKernelInvo));
        m_actKernelInvocations->setFlags(SHF_EXECINSTR);
        m_actKernelInvocations->setAddrAlign(64); // 64 default

        auto actKernelInvoWrapperRelocation = m_actKernelInvocationRela->addRelocationEntry();
        actKernelInvoWrapperRelocation->setSymbol(m_actKernelInvocationSymbol);
        actKernelInvoWrapperRelocation->setType(R_VPU_64);
        actKernelInvoWrapperRelocation->setOffset(((m_kernelsNum -1) + i) * sizeof(ActKernelInvocationWrapper) + offsetof(ActKernelInvocationWrapper, kInvo_));
        actKernelInvoWrapperRelocation->setAddend(((m_kernelsNum -1) + i) * sizeof(ActKernelInvocation));
    }

}


//*************************************

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

            if (tensorPatchingInfo.location.memLocation == MemoryLocation::ProgrammableInput) {
                relocationSection->maskFlags(VPU_SHF_JIT);
                relocationSection->maskFlags(VPU_SHF_USERINPUT);
            } else if (tensorPatchingInfo.location.memLocation == MemoryLocation::ProgrammableOutput) {
                relocationSection->maskFlags(VPU_SHF_JIT);
                relocationSection->maskFlags(VPU_SHF_USEROUTPUT);
            }
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
    if (symbolType == VPU_STT_INPUT) {
        symbolSection->maskFlags(VPU_SHF_USERINPUT);
    } else if (symbolType == VPU_STT_OUTPUT) {
        symbolSection->maskFlags(VPU_SHF_USEROUTPUT);
    }

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


    // CREATE RELA SECTION FOR ACT KERNEL PART OF MAPPED INFERENCE
    auto mappedInferenceActKernelsRelaSection = m_writer.addRelocationSection();
    mappedInferenceActKernelsRelaSection->setName(".rela.MappedInferenceActKernels");
    mappedInferenceActKernelsRelaSection->setSymbolTable(m_actKernel_symbols);
    mappedInferenceActKernelsRelaSection->setSectionToPatch(mappedInferenceSection);  


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

    //
    // Act Kernels
    //

    if (m_kernelsNum) {
        auto m_actKernelRangeWrapperSymbol = m_actKernel_symbols->addSymbolEntry();
        m_actKernelRangeWrapperSymbol->setName("actKernelRangeWrapper");
        m_actKernelRangeWrapperSymbol->setRelatedSection(m_actKernelRanges);
        m_actKernelRangeWrapperSymbol->setType(STT_SECTION);
        m_actKernelRangeWrapperSymbol->setSize(m_actKernelRanges->getDataSize());

        auto m_actKernelInvocationWrapperSymbol = m_actKernel_symbols->addSymbolEntry();
        m_actKernelInvocationWrapperSymbol->setName("actKernelInvocationWrapper");
        m_actKernelInvocationWrapperSymbol->setRelatedSection(m_actKernelInvocations);
        m_actKernelInvocationWrapperSymbol->setType(STT_SECTION);
        m_actKernelInvocationWrapperSymbol->setSize(m_actKernelInvocations->getDataSize());


        auto mappedInferenceActKernelRangeRelocation = mappedInferenceActKernelsRelaSection->addRelocationEntry();
        mappedInferenceActKernelRangeRelocation->setSymbol(m_actKernelRangeWrapperSymbol);
        mappedInferenceActKernelRangeRelocation->setType(R_VPU_64);
        mappedInferenceActKernelRangeRelocation->setOffset(offsetof(MappedInference, actKRanges) +
                                                offsetof(TaskReference<ActKernelRangeWrapper>, address));
        mappedInferenceActKernelRangeRelocation->setAddend(0);
        // segment->addSection(m_actKernelRanges);

        auto mappedInferenceActKernelInvocationRelocation = mappedInferenceActKernelsRelaSection->addRelocationEntry();
        mappedInferenceActKernelInvocationRelocation->setSymbol(m_actKernelInvocationWrapperSymbol);
        mappedInferenceActKernelInvocationRelocation->setType(R_VPU_64);
        mappedInferenceActKernelInvocationRelocation->setOffset(offsetof(MappedInference, actKInvocations) +
                                                offsetof(TaskReference<ActKernelInvocationWrapper>, address));
        mappedInferenceActKernelInvocationRelocation->setAddend(0);
        // segment->addSection(m_actKernelInvocations);
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
