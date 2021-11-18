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

#include <numeric>

using namespace vpux;
using namespace elf;
using namespace host_parsing;

VPUIP::ELFBlobSerializer::ELFBlobSerializer() {
    m_sectionSymbols = m_writer.addSymbolSection();
    m_sectionSymbols->setName(".symtab");

    m_mappedInference.leadingDmaCount[0] = 0;
    m_mappedInference.leadingDmaCount[1] = 0;

    m_mappedInference.dmaTasks[0].count = 0;
    m_mappedInference.dmaTasks[0].address = 0;

    m_mappedInference.dmaTasks[1].count = 0;
    m_mappedInference.dmaTasks[1].address = 0;

    m_mappedInference.barrierConfigs.count = 0;
    m_mappedInference.barrierConfigs.address = 0;

    m_mappedInference.variants.count = 0;
    m_mappedInference.variants.address = 0;

    m_mappedInference.invariants.count = 0;
    m_mappedInference.invariants.address = 0;

    m_mappedInference.actKInvocations.count = 0;
    m_mappedInference.actKInvocations.address = 0;

    m_mappedInference.actKRanges.count = 0;
    m_mappedInference.actKRanges.address = 0;
}

void VPUIP::ELFBlobSerializer::setDDRScratch(size_t ddrScratch) {
    m_ddrScratch = m_writer.addEmptySection();
    m_ddrScratch->setName(".ddr.Scratch");
    m_ddrScratch->setFlags(SHF_ALLOC);
    m_ddrScratch->setSize(ddrScratch);

    auto ddrScratchSymbol = m_sectionSymbols->addSymbolEntry();
    ddrScratchSymbol->setName(".ddr.Scratch");
    ddrScratchSymbol->setRelatedSection(m_ddrScratch);
    ddrScratchSymbol->setType(STT_SECTION);
    ddrScratchSymbol->setSize(ddrScratch);
    m_sectionSymbolsMapping.insert(std::make_pair(m_ddrScratch, ddrScratchSymbol));
}

void VPUIP::ELFBlobSerializer::setResourceRequirements(const host_parsing::ResourceRequirements& resourceRequirements) {
    m_resourceRequirements = resourceRequirements;
}

void VPUIP::ELFBlobSerializer::setNetworkInputs(llvm::ArrayRef<mlir::ShapedType> inputs) {
    setNetworkIO(inputs, VPU_STT_INPUT, m_networkInputSymbols, "input");
}

void VPUIP::ELFBlobSerializer::setNetworkOutputs(llvm::ArrayRef<mlir::ShapedType> outputs) {
    setNetworkIO(outputs, VPU_STT_OUTPUT, m_networkOutputSymbols, "output");
}

void VPUIP::ELFBlobSerializer::setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex) {
    m_mappedInference.leadingDmaCount[dmaEngineIndex] = leadingDMACount;
}

//************ ACT KERNELS ************

void VPUIP::ELFBlobSerializer::initActKernel(std::vector<char> elfBlob, std::string name){

    m_reader.loadElf(elfBlob.data(), elfBlob.size());

    isReaderInit = true;

    m_inputElfSecNum = m_reader.getSectionsNum();

    m_kernelName = name;

    m_kernelsNum++;

    m_actKernel_symbols = m_writer.addSymbolSection();
    m_actKernel_symbols->setName(".symtab.ActKernel");


    // Act Kernel Ranges init
    m_temp_actKernelRanges = m_writer.addBinaryDataSection<host_parsing::ActKernelRange>();
    m_temp_actKernelRanges->setName(".text." + m_kernelName + ".ActKernelRange");

    m_actKernelRangeSymbol = m_actKernel_symbols->addSymbolEntry();
    m_actKernelRangeSymbol->setName(m_kernelName + ".actKernelRange");
    m_actKernelRangeSymbol->setRelatedSection(m_temp_actKernelRanges);
    m_actKernelRangeSymbol->setType(STT_SECTION);
    m_actKernelRangeSymbol->setSize(sizeof(host_parsing::ActKernelRange));

    m_temp_actKernelRangeRela = m_writer.addRelocationSection();
    m_temp_actKernelRangeRela->setName(".rela." + m_kernelName + ".ActKernelRange");
    m_temp_actKernelRangeRela->setSymbolTable(m_actKernel_symbols);
    m_temp_actKernelRangeRela->setSectionToPatch(m_temp_actKernelRanges);

    m_actKernelRanges = m_writer.addBinaryDataSection<host_parsing::ActKernelRangeWrapper>();
    m_actKernelRanges->setName(".text." + m_kernelName + ".ActKernelRangeWrapper");

    m_actKernelRangeRela = m_writer.addRelocationSection();
    m_actKernelRangeRela->setName(".rela." + m_kernelName + ".ActKernelRangeWrapper");
    m_actKernelRangeRela->setSymbolTable(m_actKernel_symbols);
    m_actKernelRangeRela->setSectionToPatch(m_actKernelRanges);


    // Act Kernel Invocations init
    m_temp_actKernelInvocations = m_writer.addBinaryDataSection<host_parsing::ActKernelInvocation>();
    m_temp_actKernelInvocations->setName(".text." + m_kernelName + ".ActKernelInvocation");

    m_actKernelInvocationSymbol = m_actKernel_symbols->addSymbolEntry();
    m_actKernelInvocationSymbol->setName(m_kernelName + ".actKernelInvocation");
    m_actKernelInvocationSymbol->setRelatedSection(m_temp_actKernelInvocations);
    m_actKernelInvocationSymbol->setType(STT_SECTION);
    m_actKernelInvocationSymbol->setSize(sizeof(host_parsing::ActKernelInvocation));

    m_temp_actKernelInvocationRela = m_writer.addRelocationSection();
    m_temp_actKernelInvocationRela->setName(".rela." + m_kernelName + ".ActKernelInvocation");
    m_temp_actKernelInvocationRela->setSymbolTable(m_actKernel_symbols);
    m_temp_actKernelInvocationRela->setSectionToPatch(m_temp_actKernelInvocations);

    m_actKernelInvocations = m_writer.addBinaryDataSection<host_parsing::ActKernelInvocationWrapper>();
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

    for (unsigned int i = 0; i < m_inputElfSecNum; ++i) {
        auto section = m_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();

        if (strcmp(sec_name, ".text") == 0){
            temp_actKernelRange.codeSize_ = sectionHeader->sh_size;

            const auto sec_size =  sectionHeader->sh_size;
            auto sec_data = section.getData<uint8_t>();

            auto actKernelSection = m_writer.addBinaryDataSection<uint8_t>();
            actKernelSection->setName(".text." + m_kernelName);
            actKernelSection->appendData(sec_data, sec_size);
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

        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps
            temp_actKernelRange.dataSecSize_ = sectionHeader->sh_size;
        }
    }

    // ACTUAL STRUCT SERIALIZATION & RELOCS
    m_temp_actKernelRanges->appendData(&temp_actKernelRange, 1);
    m_temp_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_temp_actKernelRanges->setAddrAlign(64);

    auto actKernelRangeRelocation = m_temp_actKernelRangeRela->addRelocationEntry();
    actKernelRangeRelocation->setSymbol(actKernelSymbol);
    actKernelRangeRelocation->setType(R_VPU_32); //use 32 bit reloc
    actKernelRangeRelocation->setOffset((m_kernelsNum -1) * sizeof(host_parsing::ActKernelRange) + offsetof(host_parsing::ActKernelRange, textWindowBase_));
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

    auto dataSections = m_actKernelsMapping[textSecName];
    int inv_count = dataSections.size();

    ActKernelInvocation temp_actKernelInvo;

    // set barriers_ & kernel_args
    host_parsing::BarrierConfig dummy_barrierConfig;

    dummy_barrierConfig.group = 0;
    dummy_barrierConfig.mask = 0;
    dummy_barrierConfig.consumer_mask = 0;
    dummy_barrierConfig.producer_mask = 0;

    std::memcpy(&(temp_actKernelInvo.barriers_), &dummy_barrierConfig, sizeof(host_parsing::BarrierConfig));

    host_parsing::slf_kernel_params kernel_params;

    kernel_params.p_operation.ih = 256;
    kernel_params.p_operation.iw = 256;
    kernel_params.p_operation.ic = 1;

    auto kernelParamsSection = m_writer.addBinaryDataSection<slf_kernel_params>();
    kernelParamsSection->setName(textSecName + std::to_string(inv_count) + ".params");
    kernelParamsSection->appendData(&kernel_params, sizeof(kernel_params));
    kernelParamsSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
    kernelParamsSection->setAddrAlign(64);

    auto kernelParamsSymbol = m_actKernel_symbols->addSymbolEntry();
    kernelParamsSymbol->setName(m_kernelName + std::to_string(inv_count) + ".params");
    kernelParamsSymbol->setRelatedSection(kernelParamsSection);
    kernelParamsSymbol->setType(STT_SECTION);
    kernelParamsSymbol->setSize(kernelParamsSection->getDataSize());


    // auto tensorInSection = m_writer.addBinaryDataSection<float>();
    // tensorInSection->setName(textSecName + std::to_string(inv_count) + ".input");
    // tensorInSection->appendData(tensor_in, tensor_size);
    // tensorInSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
    // tensorInSection->setAddrAlign(64);

    // auto tensorInSymbol = m_actKernel_symbols->addSymbolEntry();
    // tensorInSymbol->setName(m_kernelName + std::to_string(inv_count) + ".input");
    // tensorInSymbol->setRelatedSection(tensorInSection);
    // tensorInSymbol->setType(STT_SECTION);
    // tensorInSymbol->setSize(tensorInSection->getDataSize());


    // auto tensorOutSection = m_writer.addBinaryDataSection<float>();
    // tensorOutSection->setName(textSecName + std::to_string(inv_count) + ".output");
    // tensorOutSection->appendData(tensor_out, tensor_size);
    // tensorOutSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
    // tensorOutSection->setAddrAlign(64);

    // auto tensorOutSymbol = m_actKernel_symbols->addSymbolEntry();
    // tensorOutSymbol->setName(m_kernelName + std::to_string(inv_count) + ".output");
    // tensorOutSymbol->setRelatedSection(tensorOutSection);
    // tensorOutSymbol->setType(STT_SECTION);
    // tensorOutSymbol->setSize(tensorOutSection->getDataSize());

    // auto kernelParamsRela = m_writer.addRelocationSection();
    // kernelParamsRela->setName(".rela." + m_kernelName + std::to_string(inv_count) + ".params");
    // kernelParamsRela->setSymbolTable(m_actKernel_symbols);
    // kernelParamsRela->setSectionToPatch(kernelParamsSection);

    // auto kernelParamsReloc_in = kernelParamsRela->addRelocationEntry();
    // kernelParamsReloc_in->setSymbol(tensorInSymbol);
    // kernelParamsReloc_in->setType(R_VPU_32);
    // kernelParamsReloc_in->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_data));
    // kernelParamsReloc_in->setAddend(0);

    // auto kernelParamsReloc_out = kernelParamsRela->addRelocationEntry();
    // kernelParamsReloc_out->setSymbol(tensorOutSymbol);
    // kernelParamsReloc_out->setType(R_VPU_32);
    // kernelParamsReloc_in->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_out));
    // kernelParamsReloc_out->setAddend(0);

    elf::writer::Symbol* actKernelDataSymbol = nullptr;

    std::string dataSecName = ".data." + m_kernelName + std::to_string(inv_count);
    std::string dataSymbolName = m_kernelName + std::to_string(inv_count);

    for (size_t i = 0; i < m_inputElfSecNum; ++i) {
        auto section = m_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();

        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps

            const auto sec_size =  sectionHeader->sh_size;
            auto sec_data = section.getData<uint8_t>();
            
            auto actKernelDataSection = m_writer.addBinaryDataSection<uint8_t>();
            actKernelDataSection->setName(dataSecName);
            actKernelDataSection->appendData(sec_data, sec_size);
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

    // create input sym
    m_networkInputSymbols = m_writer.addSymbolSection();
    m_networkInputSymbols->setName("inputs");
    m_networkInputSymbols->maskFlags(VPU_SHF_USERINPUT);

    auto inputSym = m_networkInputSymbols->addSymbolEntry();
    inputSym->setName("input");  // TODO: get name of tensor?
    inputSym->setType(VPU_STT_INPUT);
    inputSym->setValue(0);
    inputSym->setSize(256 * 256 * 1 * sizeof(float));

    auto inputRela = m_writer.addRelocationSection();
    inputRela->setName("rela.inputs");
    inputRela->setSymbolTable(m_networkInputSymbols);
    inputRela->setSectionToPatch(kernelParamsSection);
    inputRela->maskFlags(VPU_SHF_JIT);
    inputRela->maskFlags(VPU_SHF_USERINPUT);

    auto kernelParamsReloc_in = inputRela->addRelocationEntry();
    kernelParamsReloc_in->setSymbol(inputSym);
    kernelParamsReloc_in->setType(R_VPU_32);
    kernelParamsReloc_in->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_data));
    kernelParamsReloc_in->setAddend(0);


    // create output sym
    m_networkOutputSymbols = m_writer.addSymbolSection();
    m_networkOutputSymbols->setName("outputs");
    m_networkOutputSymbols->maskFlags(VPU_SHF_USEROUTPUT);

    auto outputSym = m_networkOutputSymbols->addSymbolEntry();
    outputSym->setName("output");  // TODO: get name of tensor?
    outputSym->setType(VPU_STT_OUTPUT);
    outputSym->setValue(0);
    outputSym->setSize(256 * 256 * 1 * sizeof(float));

    auto outputRela = m_writer.addRelocationSection();
    outputRela->setName("rela.outputs");
    outputRela->setSymbolTable(m_networkOutputSymbols);
    outputRela->setSectionToPatch(kernelParamsSection);
    outputRela->maskFlags(VPU_SHF_JIT);
    outputRela->maskFlags(VPU_SHF_USEROUTPUT);

    auto kernelParamsReloc_out = outputRela->addRelocationEntry();
    kernelParamsReloc_out->setSymbol(outputSym);
    kernelParamsReloc_out->setType(R_VPU_32);
    kernelParamsReloc_out->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_out));
    kernelParamsReloc_out->setAddend(0);



    // ACTUAL STRUCT SERIALIZATION & RELOCS
    // CREATE 1 SECTION, and then add data every time 
    m_temp_actKernelInvocations->appendData(&temp_actKernelInvo, 1);
    m_temp_actKernelInvocations->setFlags(SHF_EXECINSTR);
    m_temp_actKernelInvocations->setAddrAlign(64);

    auto actKernelInvoRelocation_range = m_temp_actKernelInvocationRela->addRelocationEntry();
    actKernelInvoRelocation_range->setSymbol(m_actKernelRangeSymbol);
    actKernelInvoRelocation_range->setType(R_VPU_64);
    actKernelInvoRelocation_range->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocation) 
                                                + offsetof(host_parsing::ActKernelInvocation, range_));
    actKernelInvoRelocation_range->setAddend(0);

    auto actKernelInvoRelocation_args = m_temp_actKernelInvocationRela->addRelocationEntry();
    actKernelInvoRelocation_args->setSymbol(kernelParamsSymbol);
    actKernelInvoRelocation_args->setType(R_VPU_32);
    actKernelInvoRelocation_args->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocation) 
                                                + offsetof(host_parsing::ActKernelInvocation, kernelArgs_));
    actKernelInvoRelocation_args->setAddend(0);

    auto actKernelInvoRelocation_window = m_temp_actKernelInvocationRela->addRelocationEntry();
    actKernelInvoRelocation_window->setSymbol(actKernelDataSymbol);
    actKernelInvoRelocation_window->setType(R_VPU_32); //use 32 bit reloc
    actKernelInvoRelocation_window->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocation) 
                                                + offsetof(host_parsing::ActKernelInvocation, dataWindowBase_));
    actKernelInvoRelocation_window->setAddend(0);

}

void VPUIP::ELFBlobSerializer::finalizeActKernelWrappers(){

    std::string textSecName = ".text." + m_kernelName;

    if (m_actKernelsMapping.find(textSecName) == m_actKernelsMapping.end() || m_actKernelsMapping[textSecName].size() < 1){
        std::cout<<"Error. Range or Invocations not yet set up!\n";
        return;
    }


    // RANGE
    host_parsing::ActKernelRangeWrapper final_actKernelRange;


    final_actKernelRange.kInvoCount_ = m_actKernelsMapping[textSecName].size();

    m_actKernelRanges->appendData(&final_actKernelRange, 1);
    m_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_actKernelRanges->setAddrAlign(64);

    auto actKernelRangeWrapperRelocation = m_actKernelRangeRela->addRelocationEntry();
    actKernelRangeWrapperRelocation->setSymbol(m_actKernelRangeSymbol);
    actKernelRangeWrapperRelocation->setType(R_VPU_64);
    actKernelRangeWrapperRelocation->setOffset((m_kernelsNum -1) * sizeof(host_parsing::ActKernelRangeWrapper) 
                                                + offsetof(host_parsing::ActKernelRangeWrapper, kRange_));
    actKernelRangeWrapperRelocation->setAddend((m_kernelsNum -1) * sizeof(host_parsing::ActKernelRange));

    m_mappedInference.actKRanges.count = m_kernelsNum;



    // INVOCATION
    int inv_count = m_actKernelsMapping[textSecName].size();

    host_parsing::ActKernelInvocationWrapper final_actKernelInvo;

    final_actKernelInvo.kRangeIndex_ = m_kernelsNum - 1; // set last fields of ActKernelInvoWrapper
    final_actKernelInvo.tile_ = 0;

    for (int i = 0; i < inv_count; i++){
        m_actKernelInvocations->appendData(&final_actKernelInvo, 1);
        m_actKernelInvocations->setFlags(SHF_EXECINSTR);
        m_actKernelInvocations->setAddrAlign(64); // 64 default

        auto actKernelInvoWrapperRelocation = m_actKernelInvocationRela->addRelocationEntry();
        actKernelInvoWrapperRelocation->setSymbol(m_actKernelInvocationSymbol);
        actKernelInvoWrapperRelocation->setType(R_VPU_64);
        actKernelInvoWrapperRelocation->setOffset(((m_kernelsNum -1) + i) * sizeof(host_parsing::ActKernelInvocationWrapper) 
                                                    + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_));
        actKernelInvoWrapperRelocation->setAddend(((m_kernelsNum -1) + i) * sizeof(host_parsing::ActKernelInvocation));
    }
    m_mappedInference.actKInvocations.count = m_actKernelInvocations->getDataSize() / sizeof(host_parsing::ActKernelInvocationWrapper);
    std::cout<<m_mappedInference.actKInvocations.count<<'\n';
}


//*************************************

void VPUIP::ELFBlobSerializer::setDMATasks(llvm::ArrayRef<DmaTask> dmaTasks, size_t dmaEngineIndex) {
    m_mappedInference.dmaTasks[dmaEngineIndex].count = dmaTasks.size();
    m_dmaTasks[dmaEngineIndex] = std::vector<DmaTask>(dmaTasks.begin(), dmaTasks.end());
}

void VPUIP::ELFBlobSerializer::setDPUTasks(llvm::ArrayRef<DPUTask> dpuTasks) {
    m_mappedInference.invariants.count = dpuTasks.size();
    m_mappedInference.variants.count =
            std::accumulate(dpuTasks.begin(), dpuTasks.end(), size_t{0}, [](size_t prevSum, const DPUTask& dpuTask) {
                return prevSum + dpuTask.dpuVariants.size();
            });
    m_dpuTasks = std::vector<DPUTask>(dpuTasks.begin(), dpuTasks.end());
}

void VPUIP::ELFBlobSerializer::setBarrierConfigs(llvm::ArrayRef<host_parsing::BarrierWrapper> barrierConfigs) {
    m_mappedInference.barrierConfigs.count = barrierConfigs.size();

    m_barrierConfigs = m_writer.addBinaryDataSection<BarrierWrapper>();
    m_barrierConfigs->appendData(barrierConfigs.data(), barrierConfigs.size());
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

void VPUIP::ELFBlobSerializer::setConstData(llvm::ArrayRef<uint8_t> weights) {
    m_weights = m_writer.addBinaryDataSection<uint8_t>();
    m_weights->appendData(weights.data(), weights.size());
    m_weights->setName(".data.Weights");
    m_weights->setAddrAlign(64);

    auto weightsConfigsSymbol = m_sectionSymbols->addSymbolEntry();
    weightsConfigsSymbol->setName(".weights");
    weightsConfigsSymbol->setRelatedSection(m_weights);
    weightsConfigsSymbol->setType(STT_SECTION);
    weightsConfigsSymbol->setSize(m_weights->getDataSize());
    m_sectionSymbolsMapping.insert(std::make_pair(m_weights, weightsConfigsSymbol));
}

std::vector<char> VPUIP::ELFBlobSerializer::getBlob() {
    finalize();
    return m_writer.generateELF();
}

void VPUIP::ELFBlobSerializer::setNetworkIO(llvm::ArrayRef<mlir::ShapedType> inputsOrOutputs, uint8_t symbolType,
                                            writer::SymbolSection*& symbolSection, const std::string& symbolName) {
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
    // host_parsing::MappedInference
    //

    auto mappedInferenceSection = m_writer.addBinaryDataSection<MappedInference>();
    mappedInferenceSection->setName(".text.MappedInference");
    mappedInferenceSection->appendData(m_mappedInference);
    mappedInferenceSection->setFlags(SHF_EXECINSTR);
    mappedInferenceSection->setAddrAlign(64);

    auto mappedInferenceSymbol = m_sectionSymbols->addSymbolEntry();
    mappedInferenceSymbol->setName(".ddr.mappedInference");
    mappedInferenceSymbol->setRelatedSection(mappedInferenceSection);
    mappedInferenceSymbol->setType(STT_SECTION);
    mappedInferenceSymbol->setSize(mappedInferenceSection->getDataSize());


    auto mappedInferenceRelaSection = m_writer.addRelocationSection();
    mappedInferenceRelaSection->setName(".rela.host_parsing::MappedInference");
    mappedInferenceRelaSection->setSymbolTable(m_sectionSymbols);
    mappedInferenceRelaSection->setSectionToPatch(mappedInferenceSection);

    auto entryPoint = m_sectionSymbols->addSymbolEntry();
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

    finalizeDMA();
    for (size_t i = 0; i < m_dmaTasksSections.size(); ++i) {
        const auto& dmaTasks = m_dmaTasksSections[i];
        if (dmaTasks) {
            auto mappedInferenceDMARelocation = mappedInferenceRelaSection->addRelocationEntry();
            mappedInferenceDMARelocation->setSymbol(m_sectionSymbolsMapping.at(dmaTasks));
            mappedInferenceDMARelocation->setType(R_VPU_64);
            mappedInferenceDMARelocation->setOffset(
                    offsetof(host_parsing::MappedInference, dmaTasks) + sizeof(dmaTasks) * i +
                    offsetof(host_parsing::TaskReference<host_parsing::DmaWrapper>, address));
            mappedInferenceDMARelocation->setAddend(0);
            segment->addSection(dmaTasks);
        }
    }

    //
    // DPUTasks
    //

    finalizeDPU();
    if (m_dpuInvariants) {
        VPUX_THROW_UNLESS(m_dpuVariants, "DPU variants can't be NULL");

        auto invariantsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        invariantsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_dpuInvariants));
        invariantsRelocation->setType(R_VPU_64);
        invariantsRelocation->setOffset(
                offsetof(host_parsing::MappedInference, invariants) +
                offsetof(host_parsing::TaskReference<host_parsing::DPUInvariantWrapper>, address));
        invariantsRelocation->setAddend(0);
        segment->addSection(m_dpuInvariants);

        auto variantsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        variantsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_dpuVariants));
        variantsRelocation->setType(R_VPU_64);
        variantsRelocation->setOffset(offsetof(host_parsing::MappedInference, variants) +
                                      offsetof(host_parsing::TaskReference<host_parsing::DPUVariantWrapper>, address));
        variantsRelocation->setAddend(0);
        segment->addSection(m_dpuVariants);
    }

    //
    // BarrierConfigs
    //

    if (m_barrierConfigs) {
        auto mappedInferenceBarrierConfigsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceBarrierConfigsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_barrierConfigs));
        mappedInferenceBarrierConfigsRelocation->setType(R_VPU_64);
        mappedInferenceBarrierConfigsRelocation->setOffset(
                offsetof(host_parsing::MappedInference, barrierConfigs) +
                offsetof(host_parsing::TaskReference<host_parsing::DmaWrapper>, address));
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
        mappedInferenceActKernelRangeRelocation->setOffset(offsetof(host_parsing::MappedInference, actKRanges) +
                                                offsetof(TaskReference<ActKernelRangeWrapper>, address));
        mappedInferenceActKernelRangeRelocation->setAddend(0);
        // segment->addSection(m_actKernelRanges);

        auto mappedInferenceActKernelInvocationRelocation = mappedInferenceActKernelsRelaSection->addRelocationEntry();
        mappedInferenceActKernelInvocationRelocation->setSymbol(m_actKernelInvocationWrapperSymbol);
        mappedInferenceActKernelInvocationRelocation->setType(R_VPU_64);
        mappedInferenceActKernelInvocationRelocation->setOffset(offsetof(host_parsing::MappedInference, actKInvocations) +
                                                offsetof(TaskReference<ActKernelInvocationWrapper>, address));
        mappedInferenceActKernelInvocationRelocation->setAddend(0);
        // segment->addSection(m_actKernelInvocations);
    }

}

void VPUIP::ELFBlobSerializer::finalizeDMA() {
    for (size_t dmaEngineIndex = 0; dmaEngineIndex < m_dmaTasks.size(); ++dmaEngineIndex) {
        auto& dmaTasks = m_dmaTasks[dmaEngineIndex];
        if (dmaTasks.empty()) {
            continue;
        }

        auto& dmaTasksSection = m_dmaTasksSections[dmaEngineIndex];
        dmaTasksSection = m_writer.addBinaryDataSection<host_parsing::DmaWrapper>();
        dmaTasksSection->setName(".text.DMATasks" + std::to_string(dmaEngineIndex));
        dmaTasksSection->setFlags(SHF_EXECINSTR);
        dmaTasksSection->setAddrAlign(64);

        auto dmaTasksSymbol = m_sectionSymbols->addSymbolEntry();
        dmaTasksSymbol->setName(".ddr.dmaTasks" + std::to_string(dmaEngineIndex));
        dmaTasksSymbol->setRelatedSection(dmaTasksSection);
        dmaTasksSymbol->setType(STT_SECTION);
        m_sectionSymbolsMapping.insert(std::make_pair(dmaTasksSection, dmaTasksSymbol));

        RelocationManager relocationManager(dmaTasksSection, ".rela.DMA", *this);

        for (size_t i = 0; i < dmaTasks.size(); ++i) {
            auto& dmaTask = dmaTasks[i];

            const auto transactionOffset =
                    i * sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction);
            relocationManager.addRelocation(dmaTask.input, R_VPU_64,
                                            transactionOffset + offsetof(host_parsing::DmaDescriptor, src));
            relocationManager.addRelocation(dmaTask.output, R_VPU_64,
                                            transactionOffset + offsetof(host_parsing::DmaDescriptor, dst));

            if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::DDR_DMA) {
                relocationManager.addRelocation(m_sectionSymbols, dmaTasksSymbol, R_VPU_64_OR,
                                                dmaTask.linkAddress.dmaTaskIndex * sizeof(host_parsing::DmaWrapper) +
                                                        offsetof(host_parsing::DmaWrapper, transaction),
                                                transactionOffset);
            } else if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::RTM_DMA) {
                dmaTask.dmaDescriptor.transaction.link_address = dmaTask.linkAddress.dmaTaskIndex;
                relocationManager.addRelocation(NNRD_SYM_RTM_DMA0 + dmaEngineIndex, R_VPU_64_OR_RTM,
                                                sizeof(host_parsing::DmaWrapper), transactionOffset);
            }

            if (dmaTask.dmaDescriptor.transaction.barriers.prod_mask) {
                relocationManager.addRelocation(
                        NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                        transactionOffset + offsetof(DmaDescriptor, barriers) + offsetof(DmaBarrierCfg, prod_mask));
            }

            if (dmaTask.dmaDescriptor.transaction.barriers.cons_mask) {
                relocationManager.addRelocation(
                        NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                        transactionOffset + offsetof(DmaDescriptor, barriers) + offsetof(DmaBarrierCfg, cons_mask));
            }
        }

        std::vector<host_parsing::DmaWrapper> dmaDescriptors;
        dmaDescriptors.reserve(dmaTasks.size());
        for (const auto& dmaTask : dmaTasks) {
            dmaDescriptors.push_back(dmaTask.dmaDescriptor);
        }
        dmaTasksSection->appendData(dmaDescriptors.data(), dmaDescriptors.size());
        dmaTasksSymbol->setSize(dmaTasksSection->getDataSize());
    }
}

void VPUIP::ELFBlobSerializer::finalizeDPU() {
    if (m_dpuTasks.empty()) {
        return;
    }

    m_dpuInvariants = m_writer.addBinaryDataSection<host_parsing::DPUInvariantWrapper>();
    m_dpuInvariants->setName(".text.Invariants");
    m_dpuInvariants->setFlags(SHF_EXECINSTR);
    m_dpuInvariants->setAddrAlign(64);

    auto invariantsSymbol = m_sectionSymbols->addSymbolEntry();
    invariantsSymbol->setName(".ddr.Invariants");
    invariantsSymbol->setRelatedSection(m_dpuInvariants);
    invariantsSymbol->setType(STT_SECTION);
    m_sectionSymbolsMapping.insert(std::make_pair(m_dpuInvariants, invariantsSymbol));

    m_dpuVariants = m_writer.addBinaryDataSection<host_parsing::DPUVariantWrapper>();
    m_dpuVariants->setName(".text.Variants");
    m_dpuVariants->setFlags(SHF_EXECINSTR);
    m_dpuVariants->setAddrAlign(64);

    auto variantsSymbol = m_sectionSymbols->addSymbolEntry();
    variantsSymbol->setName(".ddr.Variants");
    variantsSymbol->setRelatedSection(m_dpuVariants);
    variantsSymbol->setType(STT_SECTION);
    m_sectionSymbolsMapping.insert(std::make_pair(m_dpuVariants, variantsSymbol));

    RelocationManager invariantRelocationManager(m_dpuInvariants, ".rela.Inv", *this);
    RelocationManager variantRelocationManager(m_dpuVariants, ".rela.Var", *this);
    size_t variantIndex = 0;
    for (size_t i = 0; i < m_dpuTasks.size(); i++) {
        auto& dpuTask = m_dpuTasks[i];

        updateInvariant(dpuTask.dpuInvariant, invariantRelocationManager,
                        i * sizeof(DPUInvariantWrapper) + offsetof(DPUInvariantWrapper, invariant));

        const auto barriersOffset = i * sizeof(DPUInvariantWrapper) + offsetof(DPUInvariantWrapper, invariant) +
                                    offsetof(DPUInvariant, barriers);
        if (dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers.producer_mask) {
            invariantRelocationManager.addRelocation(NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                                                     barriersOffset + offsetof(BarrierConfig, producer_mask));
        }

        if (dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers.consumer_mask) {
            invariantRelocationManager.addRelocation(NNRD_SYM_BARRIERS_START, R_VPU_64_LSHIFT, 0,
                                                     barriersOffset + offsetof(BarrierConfig, consumer_mask));
        }

        // TODO: this is purely an optimization so it should work w/o it
        dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers.group = 0;
        dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers.mask = 0;
        // reduce_wait_mask_to_8bit(dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers);

        for (auto& variant : m_dpuTasks[i].dpuVariants) {
            variant.dpuVariantWrapper.variant.invariant_addr = variant.dpuVariantWrapper.invariant_index;
            variantRelocationManager.addRelocation(NNRD_SYM_RTM_IVAR, R_VPU_32_OR_RTM,
                                                   sizeof(host_parsing::DPUInvariantWrapper),
                                                   variantIndex * sizeof(host_parsing::DPUVariantWrapper) +
                                                           offsetof(host_parsing::DPUVariantWrapper, variant) +
                                                           offsetof(host_parsing::DPUVariant, invariant_addr));
            variantRelocationManager.addRelocation(m_dpuTasks[i].dpuInvariant.weightsTable, R_VPU_32_SUM,
                                                   variantIndex * sizeof(DPUVariantWrapper) +
                                                           offsetof(DPUVariantWrapper, variant) +
                                                           offsetof(DPUVariant, weight_table_offset));
            variant.dpuVariantWrapper.invariant_index = i;
            variantIndex++;
        }
    }

    std::vector<host_parsing::DPUInvariantWrapper> dpuInvariantWrappers;
    std::vector<host_parsing::DPUVariantWrapper> dpuVariantWrappers;
    dpuInvariantWrappers.reserve(m_mappedInference.invariants.count);
    dpuVariantWrappers.reserve(m_mappedInference.variants.count);
    for (const auto& dpuTask : m_dpuTasks) {
        dpuInvariantWrappers.push_back(dpuTask.dpuInvariant.dpuInvariantWrapper);
        for (const auto& dpuVariant : dpuTask.dpuVariants) {
            dpuVariantWrappers.push_back(dpuVariant.dpuVariantWrapper);
        }
    }

    m_dpuInvariants->appendData(dpuInvariantWrappers.data(), dpuInvariantWrappers.size());
    invariantsSymbol->setSize(m_dpuInvariants->getDataSize());

    m_dpuVariants->appendData(dpuVariantWrappers.data(), dpuVariantWrappers.size());
    variantsSymbol->setSize(m_dpuVariants->getDataSize());
}

void VPUIP::ELFBlobSerializer::updateInvariant(DPUInvariantTask& invariantTask, RelocationManager& relocationManager,
                                               uint64_t invariantSectionOffset) {
    // Hardcoding to 3 for now - matches POC.
    // FIXME: update this for MTL.
    // Assuming 3 is from:
    /*
     * For MeteorLake integration, three contexts can execute simultaneously
     * (two compute engines, each on separate NCE Slices and a copy function on a third context).
     * -- VPU2.7 HAS
     */
    constexpr unsigned int numSlices = 3;
    updateInvariantSOH(invariantTask, relocationManager, invariantSectionOffset);

    const auto input = invariantTask.input;

    for (size_t i = 0; i < 4; ++i) {
        relocationManager.addRelocation(input, R_VPU_32,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, act_offset) +
                                                i * sizeof(host_parsing::DPUInvariantRegisters::act_offset[0]));
    }

    auto& invariant = invariantTask.dpuInvariantWrapper.invariant;

    invariant.registers.se_sp_addr[1].se_addr = ((1 * SLICE_LENGTH) >> 4);
    invariant.registers.se_sp_addr[2].se_addr = ((2 * SLICE_LENGTH) >> 4);
    invariant.registers.se_sp_addr[3].se_addr = ((3 * SLICE_LENGTH) >> 4);

    // FIXME: hardcoded and directly copied from POC runtime...
    invariant.registers.base_offset_a = 0x200;
    invariant.registers.base_offset_b = 0x602;

    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
        const auto seSpAddr = invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                              offsetof(host_parsing::DPUInvariantRegisters, se_sp_addr) +
                              sizeof(host_parsing::DPUInvariantRegisters::se_sp_addr[0]);

        relocationManager.addRelocation(input, R_VPU_32, seSpAddr, OffsetToUse::SPARSITY_TABLE);
        relocationManager.addRelocation(input, R_VPU_32,
                                        seSpAddr + sizeof(host_parsing::DPUInvariantRegisters::se_sp_addr[0].se_addr),
                                        OffsetToUse::SPARSITY_MAP);
    }

    for (size_t i = 0; i < 4; ++i) {
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_BASE_SUB,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, base_adr) +
                                                i * sizeof(host_parsing::DPUInvariantRegisters::base_adr));
    }

    for (unsigned int i = 0; i < numSlices; ++i) {
        invariant.registers.odu_cast[i].odu_cast_bf.cast_enable = i;
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_OFFSET_CMP_OR,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, odu_cast) +
                                                i * sizeof(host_parsing::DPUInvariantRegisters::odu_cast[0]));

        invariant.registers.odu_cast[i].odu_cast_bf.cast_offset = i;
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_OFFSET_4_BIT_SHIFT_OR,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, odu_cast) +
                                                i * sizeof(host_parsing::DPUInvariantRegisters::odu_cast[0]));
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_pt) {
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_BASE,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, pt_base),
                                        OffsetToUse::SPARSITY_TABLE);
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_sp) {
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_BASE,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, sp_base),
                                        OffsetToUse::SPARSITY_MAP);
    }

    relocationManager.addRelocation(NNRD_SYM_FIFO_BASE, R_VPU_32, invariantTask.dpuInvariantWrapper.cluster,
                                    invariantSectionOffset * offsetof(host_parsing::DPUInvariantWrapper, cluster));

    switch (invariantTask.opType) {
    case vpux::VPUIP::NCETaskType::CONV:
    case vpux::VPUIP::NCETaskType::CMCONV:
    case vpux::VPUIP::NCETaskType::MAXPOOL:
    case vpux::VPUIP::NCETaskType::AVEPOOL:
    case vpux::VPUIP::NCETaskType::DWCONV: {
        relocationManager.addRelocation(
                invariantTask.weights, R_VPU_32,
                invariantSectionOffset + offsetof(DPUInvariant, registers) + offsetof(DPUInvariantRegisters, wt_offset),
                OffsetToUse::BASE);  // TODO: Why we need this base?

        relocationManager.addRelocation(invariantTask.weightsTable, R_VPU_32,
                                        invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                                offsetof(host_parsing::DPUInvariantRegisters, weight_start));

        switch (invariantTask.opType) {
        case vpux::VPUIP::NCETaskType::DWCONV:
        case vpux::VPUIP::NCETaskType::CMCONV:
        case vpux::VPUIP::NCETaskType::MAXPOOL:
            invariant.registers.tensor_start = 0;
            break;
        default:
            break;
        }
        break;
    }

    default:
        VPUX_THROW("Layer type {0} is not supported", invariantTask.opType);
        break;
    }
}

void VPUIP::ELFBlobSerializer::updateInvariantSOH(DPUInvariantTask& invariantTask, RelocationManager& relocationManager,
                                                  uint64_t invariantSectionOffset) {
    const auto& invariant = invariantTask.dpuInvariantWrapper.invariant;
    auto& input = invariantTask.input;
    if (invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment) {
        // Split over H segmenting
        input.location.locationIndex = 1;

        for (int i = 0; i < 3; i++) {
            if (invariant.registers.se_sp_size[i].se_sp_size_bf.se_seg_size) {
                input.location.locationIndex = 1 << (i + 1);
                const auto seSpAddr = invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                      offsetof(host_parsing::DPUInvariantRegisters, se_sp_addr) +
                                      (i + 1) * sizeof(host_parsing::DPUInvariantRegisters::se_sp_addr[0]);

                if (invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                    relocationManager.addRelocation(input, R_VPU_32_SUM, seSpAddr);
                } else {
                    // HW issue (A0): se_addr for segments 2+ need and offset from the real address of the segment.
                    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                        relocationManager.addRelocation(input, R_VPU_32, seSpAddr, OffsetToUse::SPARSITY_TABLE);
                    }

                    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                        relocationManager.addRelocation(
                                input, R_VPU_32,
                                seSpAddr + sizeof(host_parsing::DPUInvariantRegisters::se_sp_addr[0].se_addr),
                                OffsetToUse::SPARSITY_MAP);
                    }

                    // Previous layers have set the ODU base select to the cluster index
                    // Need to have matching logic at IDU side
                    relocationManager.addRelocation(
                            input, R_VPU_32,
                            invariantSectionOffset + offsetof(host_parsing::DPUInvariant, registers) +
                                    offsetof(host_parsing::DPUInvariantRegisters, act_offset) +
                                    (i + 1) * sizeof(host_parsing::DPUInvariantRegisters::act_offset[0]));
                }
            }
        }

        input.location.locationIndex = 1;
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
                                                                Elf_Word type, Elf64_Addr sectionOffset,
                                                                OffsetToUse offsetToUse) {
    uint64_t offset = 0;
    switch (offsetToUse) {
    case OffsetToUse::DATA:
        offset = tensorPatchingInfo.dataOffset;
        break;
    case OffsetToUse::SPARSITY_MAP:
        offset = tensorPatchingInfo.sparsityMapOffset;
        break;
    case OffsetToUse::SPARSITY_TABLE:
        offset = tensorPatchingInfo.sparsityTableOffset;
        break;
    default:
        offset = 0;
        break;
    }

    switch (tensorPatchingInfo.location.memLocation) {
    case VPUIP::MemoryLocation::ProgrammableInput:
        addRelocation(
                getRelocationSection(m_elfBlobSerializer.m_networkInputSymbols,
                                     tensorPatchingInfo.location.memLocation),
                m_elfBlobSerializer.m_networkInputSymbols->getSymbols()[tensorPatchingInfo.location.locationIndex + 1]
                        .get(),
                type, offset, sectionOffset);
        break;
    case VPUIP::MemoryLocation::ProgrammableOutput:
        addRelocation(
                getRelocationSection(m_elfBlobSerializer.m_networkOutputSymbols,
                                     tensorPatchingInfo.location.memLocation),
                m_elfBlobSerializer.m_networkOutputSymbols->getSymbols()[tensorPatchingInfo.location.locationIndex + 1]
                        .get(),
                type, offset, sectionOffset);
        break;
    case VPUIP::MemoryLocation::VPU_DDR_Heap:
        addRelocation(
                getRelocationSection(m_elfBlobSerializer.m_sectionSymbols, tensorPatchingInfo.location.memLocation),
                m_elfBlobSerializer.m_sectionSymbolsMapping.at(m_elfBlobSerializer.m_ddrScratch), type, offset,
                sectionOffset);
        break;
    case VPUIP::MemoryLocation::VPU_CMX_NN:
        addRelocation(NNRD_SYM_NNCXM_SLICE_BASE_ADDR, type, offset, sectionOffset);
        break;
    case VPUIP::MemoryLocation::GraphFile:
        addRelocation(
                getRelocationSection(m_elfBlobSerializer.m_sectionSymbols, tensorPatchingInfo.location.memLocation),
                m_elfBlobSerializer.m_sectionSymbolsMapping.at(m_elfBlobSerializer.m_weights), type,
                tensorPatchingInfo.dataOffset, sectionOffset);
        break;
    default:
        VPUX_THROW("Unsupported MemoryLocation {}", tensorPatchingInfo.location.memLocation);
        break;
    }
}

void VPUIP::ELFBlobSerializer::RelocationManager::addRelocation(const writer::SymbolSection* symbolSection,
                                                                const writer::Symbol* symbol, Elf_Word type,
                                                                Elf_Sxword addend, Elf64_Addr sectionOffset) {
    addRelocation(getRelocationSection(symbolSection), symbol, type, addend, sectionOffset);
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

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::getRelocationSection(
        const elf::writer::SymbolSection* symbolSection) {
    if (m_symbolTableToRelocation.find(symbolSection) == m_symbolTableToRelocation.end()) {
        auto relocationTable = createRelocationSection(symbolSection);
        m_symbolTableToRelocation[symbolSection] = relocationTable;
    }

    return m_symbolTableToRelocation.at(symbolSection);
}

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::getRelocationSection(
        const elf::writer::SymbolSection* symbolSection, VPUIP::MemoryLocation memoryLocation) {
    auto relocationTable = getRelocationSection(symbolSection);

    if (memoryLocation == VPUIP::MemoryLocation::ProgrammableInput) {
        relocationTable->maskFlags(VPU_SHF_JIT);
        relocationTable->maskFlags(VPU_SHF_USERINPUT);
    } else if (memoryLocation == VPUIP::MemoryLocation::ProgrammableOutput) {
        relocationTable->maskFlags(VPU_SHF_JIT);
        relocationTable->maskFlags(VPU_SHF_USEROUTPUT);
    }

    return relocationTable;
}

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::createRelocationSection(
        const writer::SymbolSection* symbolSection) {
    auto relocationTable = m_elfBlobSerializer.m_writer.addRelocationSection();
    relocationTable->setName(m_relocationSectionName);
    relocationTable->setSectionToPatch(m_sectionToPatch);
    relocationTable->setSymbolTable(symbolSection);
    return relocationTable;
}
