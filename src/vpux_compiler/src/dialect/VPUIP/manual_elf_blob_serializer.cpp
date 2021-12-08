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

#include <vpux/compiler/dialect/VPUIP/manual_elf_blob_serializer.hpp>

#include <vpux/compiler/utils/types.hpp>

#include <elf/types/vpu_extensions.hpp>

#include <numeric>

using namespace vpux;
using namespace elf;
using namespace host_parsing;

VPUIP::ManualELFBlobSerializer::ManualELFBlobSerializer() {
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

// void VPUIP::ManualELFBlobSerializer::setDDRScratch(size_t ddrScratch) {
//     m_ddrScratch = m_writer.addEmptySection();
//     m_ddrScratch->setName(".ddr.Scratch");
//     m_ddrScratch->setFlags(SHF_ALLOC);
//     m_ddrScratch->setSize(ddrScratch);

//     auto ddrScratchSymbol = m_sectionSymbols->addSymbolEntry();
//     ddrScratchSymbol->setName(".ddr.Scratch");
//     ddrScratchSymbol->setRelatedSection(m_ddrScratch);
//     ddrScratchSymbol->setType(STT_SECTION);
//     ddrScratchSymbol->setSize(ddrScratch);
//     m_sectionSymbolsMapping.insert(std::make_pair(m_ddrScratch, ddrScratchSymbol));
// }

// void VPUIP::ManualELFBlobSerializer::setResourceRequirements(const host_parsing::ResourceRequirements& resourceRequirements) {
//     m_resourceRequirements = resourceRequirements;
// }

// void VPUIP::ManualELFBlobSerializer::setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex) {
//     m_mappedInference.leadingDmaCount[dmaEngineIndex] = leadingDMACount;
// }

//************ ACT KERNELS ************

void VPUIP::ManualELFBlobSerializer::initActKernel(std::vector<char> elfBlob, std::string name){

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
    m_actKernelInvocations = m_writer.addBinaryDataSection<host_parsing::ActKernelInvocationWrapper>();
    m_actKernelInvocations->setName(".text." + m_kernelName + ".ActKernelInvocationWrapper");

    m_actKernelInvocationRela = m_writer.addRelocationSection();
    m_actKernelInvocationRela->setName(".rela." + m_kernelName + ".ActKernelInvocationWrapper");
    m_actKernelInvocationRela->setSymbolTable(m_actKernel_symbols);
    m_actKernelInvocationRela->setSectionToPatch(m_actKernelInvocations);

    m_actKernelInvocationSpecialRela = m_writer.addRelocationSection();
    m_actKernelInvocationSpecialRela->setName(".rela." + m_kernelName + ".ActKernelInvocationWrapper_special");
    m_actKernelInvocationSpecialRela->setSpecialSymbolTable(VPU_RT_SYMTAB);
    m_actKernelInvocationSpecialRela->setSectionToPatch(m_actKernelInvocations);
}


void VPUIP::ManualELFBlobSerializer::addActKernel(){

    if (!isReaderInit){
        std::cout<<"READER NOT INIT\n";
        return;
    }

    ActKernelRange temp_actKernelRange;
    host_parsing::ActKernelRangeWrapper final_actKernelRange;

    elf::writer::Symbol* actKernelSymbol = nullptr;

    // set static fields of actKernelRange backend
    const ELF32Header* actKernelHeader = m_reader.getHeader();
    auto my_actKernelEntry = actKernelHeader->e_entry;

    temp_actKernelRange.kernelEntry_ = my_actKernelEntry;

    // get .text section and move it to new elf
    for (unsigned int i = 0; i < m_inputElfSecNum; ++i) {
        auto section = m_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();

        if (strcmp(sec_name, ".text") == 0){
            const auto sec_size =  sectionHeader->sh_size;

            // set type_ & codeSize_
            temp_actKernelRange.type_ = ActWLType::WL_KERNEL;
            temp_actKernelRange.codeSize_ = sec_size;

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

        // get data size to set it in actKernelRange backend
        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps
            temp_actKernelRange.dataSecSize_ = sectionHeader->sh_size;
        }
    }

    m_temp_actKernelRanges->appendData(&temp_actKernelRange, 1);
    m_temp_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_temp_actKernelRanges->setAddrAlign(64);

    auto actKernelRangeRelocation = m_temp_actKernelRangeRela->addRelocationEntry();
    actKernelRangeRelocation->setSymbol(actKernelSymbol);
    actKernelRangeRelocation->setType(R_VPU_32);
    actKernelRangeRelocation->setOffset((m_kernelsNum -1) * sizeof(ActKernelRange) + offsetof(ActKernelRange, textWindowBase_));
    actKernelRangeRelocation->setAddend(0);

    final_actKernelRange.kRange_ = temp_actKernelRange;
    final_actKernelRange.kInvoCount_ = 1;

    m_actKernelRanges->appendData(&final_actKernelRange, 1);
    m_actKernelRanges->setFlags(SHF_EXECINSTR);
    m_actKernelRanges->setAddrAlign(64);

    auto textWindow_reloc = m_actKernelRangeRela->addRelocationEntry();
    textWindow_reloc->setSymbol(actKernelSymbol);
    textWindow_reloc->setType(R_VPU_32);
    textWindow_reloc->setOffset((m_kernelsNum -1) * sizeof(host_parsing::ActKernelRangeWrapper) 
                                                + offsetof(host_parsing::ActKernelRangeWrapper, kRange_)
                                                + offsetof(host_parsing::ActKernelRange, textWindowBase_));
    textWindow_reloc->setAddend(0);
}

void VPUIP::ManualELFBlobSerializer::addActInvocation(){

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

    host_parsing::ActKernelInvocation temp_actKernelInvo;
    host_parsing::ActKernelInvocationWrapper final_actKernelInvo;

    // set barriers_ & kernel_args for actKernelInvo backend
    host_parsing::BarrierConfig dummy_barrierConfig;

    dummy_barrierConfig.group = 0;
    dummy_barrierConfig.mask = 0;
    dummy_barrierConfig.consumer_mask = 0b0000000000000001;
    dummy_barrierConfig.producer_mask = 0b0000000000000010;

    temp_actKernelInvo.barriers_ = dummy_barrierConfig;

    host_parsing::slf_kernel_params kernel_params;

    kernel_params.p_operation.ih = IO_WIDTH;
    kernel_params.p_operation.iw = IO_HEIGHT;
    kernel_params.p_operation.ic = IO_CHANNELS;

    auto kernelParamsSection = m_writer.addBinaryDataSection<slf_kernel_params>();
    kernelParamsSection->setName(textSecName + std::to_string(inv_count) + ".params");
    kernelParamsSection->appendData(&kernel_params, 1);
    kernelParamsSection->setFlags(SHF_EXECINSTR + SHF_ALLOC);
    kernelParamsSection->setAddrAlign(64);

    auto kernelParamsSymbol = m_actKernel_symbols->addSymbolEntry();
    kernelParamsSymbol->setName(m_kernelName + std::to_string(inv_count) + ".params");
    kernelParamsSymbol->setRelatedSection(kernelParamsSection);
    kernelParamsSymbol->setType(STT_SECTION);
    kernelParamsSymbol->setSize(kernelParamsSection->getDataSize());

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

    // create IO relocs

    auto ioRela = m_writer.addRelocationSection();
    ioRela->setName(".rela.param_IO");
    ioRela->setSpecialSymbolTable(VPU_RT_SYMTAB);
    ioRela->setSectionToPatch(kernelParamsSection);

    auto kernelParamsReloc_in = ioRela->addRelocationEntry();
    kernelParamsReloc_in->setSpecialSymbol(NNRD_SYM_NNCXM_SLICE_BASE_ADDR);
    kernelParamsReloc_in->setType(R_VPU_32);
    kernelParamsReloc_in->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_data));
    kernelParamsReloc_in->setAddend(0);

    auto kernelParamsReloc_out = ioRela->addRelocationEntry();
    kernelParamsReloc_out->setSpecialSymbol(NNRD_SYM_NNCXM_SLICE_BASE_ADDR);
    kernelParamsReloc_out->setType(R_VPU_32);
    kernelParamsReloc_out->setOffset(offsetof(host_parsing::slf_kernel_params, p_act_out));
    kernelParamsReloc_out->setAddend(1000);


    // setup actKernelInvo wrapper fields
    final_actKernelInvo.kInvo_ = temp_actKernelInvo;
    final_actKernelInvo.kRangeIndex_ = m_kernelsNum - 1;
    final_actKernelInvo.tile_ = 0;
    final_actKernelInvo.start_after_ = 0;
    final_actKernelInvo.clean_after_ = 1;

    m_actKernelInvocations->appendData(&final_actKernelInvo, 1);
    m_actKernelInvocations->setFlags(SHF_EXECINSTR);
    m_actKernelInvocations->setAddrAlign(64); // 64 default

    auto range_reloc = m_actKernelInvocationRela->addRelocationEntry();
    range_reloc->setSymbol(m_actKernelRangeSymbol);
    range_reloc->setType(R_VPU_32);
    range_reloc->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocationWrapper)
                                                + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) 
                                                + offsetof(host_parsing::ActKernelInvocation, range_));
    range_reloc->setAddend((m_kernelsNum -1) * sizeof(host_parsing::ActKernelRange));

    auto args_reloc = m_actKernelInvocationRela->addRelocationEntry();
    args_reloc->setSymbol(kernelParamsSymbol);
    args_reloc->setType(R_VPU_32);
    args_reloc->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocationWrapper)
                                                + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) 
                                                + offsetof(host_parsing::ActKernelInvocation, kernelArgs_));
    args_reloc->setAddend(0);

    auto dataWindow_reloc = m_actKernelInvocationRela->addRelocationEntry();
    dataWindow_reloc->setSymbol(actKernelDataSymbol);
    dataWindow_reloc->setType(R_VPU_32);
    dataWindow_reloc->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocationWrapper)
                                                + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) 
                                                + offsetof(host_parsing::ActKernelInvocation, dataWindowBase_));
    dataWindow_reloc->setAddend(0);

    auto prod_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
    prod_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
    prod_mask_reloc->setType(R_VPU_64_LSHIFT);
    prod_mask_reloc->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocationWrapper)
                                + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_)
                                + offsetof(host_parsing::ActKernelInvocation, barriers_)
                                + offsetof(host_parsing::BarrierConfig, producer_mask));
    prod_mask_reloc->setAddend(0);

    auto cons_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
    cons_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
    cons_mask_reloc->setType(R_VPU_64_LSHIFT);
    cons_mask_reloc->setOffset(((m_kernelsNum -1) + inv_count) * sizeof(host_parsing::ActKernelInvocationWrapper)
                                + offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_)
                                + offsetof(host_parsing::ActKernelInvocation, barriers_)
                                + offsetof(host_parsing::BarrierConfig, consumer_mask));
    cons_mask_reloc->setAddend(0);
}

host_parsing::ActKernelRuntimeConfigs VPUIP::ManualELFBlobSerializer::setActRtConfigs(){

    host_parsing::ActKernelRuntimeConfigs actKernelConfigs;

    actKernelConfigs.stackFrames_[0] = 0x2E000800;
    actKernelConfigs.stackFrames_[1] = 0x2E000C00;
    actKernelConfigs.stackFrames_[2] = 0x2E200800;
    actKernelConfigs.stackFrames_[3] = 0x2E200C00;

    actKernelConfigs.stackSize_ = 16384; // 16 kB

    actKernelConfigs.useScheduleEmbeddedRt_ = false;

    // create 1MB empty section
    actKernelConfigs.codeWindowBufferSize_ = 1048576 * sizeof(uint8_t);

    auto actRtWindow_sec = m_writer.addEmptySection();
    actRtWindow_sec->setName(".actRt");
    actRtWindow_sec->setSize(1048576 * sizeof(uint8_t));
    actRtWindow_sec->setFlags(SHF_EXECINSTR + SHF_ALLOC);
    actRtWindow_sec->setAddrAlign(1024);

    m_actRtConfigMainSymbol = m_actKernel_symbols->addSymbolEntry();
    m_actRtConfigMainSymbol->setName("actRt");
    m_actRtConfigMainSymbol->setRelatedSection(actRtWindow_sec);
    m_actRtConfigMainSymbol->setType(STT_SECTION);
    m_actRtConfigMainSymbol->setSize(actRtWindow_sec->getDataSize());

    return actKernelConfigs;
}

//*************************************
// DMA

void VPUIP::ManualELFBlobSerializer::initCmxDMA() {
    m_dmaTasksSection = m_writer.addBinaryDataSection<host_parsing::DmaWrapper>();
    m_dmaTasksSection->setName(".text.DMATasks");
    m_dmaTasksSection->setFlags(SHF_EXECINSTR);
    m_dmaTasksSection->setAddrAlign(64);

    m_dmaTasksSymbol = m_sectionSymbols->addSymbolEntry();
    m_dmaTasksSymbol->setName("dmaTasks");
    m_dmaTasksSymbol->setRelatedSection(m_dmaTasksSection);
    m_dmaTasksSymbol->setType(STT_SECTION);
    m_sectionSymbolsMapping.insert(std::make_pair(m_dmaTasksSection, m_dmaTasksSymbol));

    m_dmaTasksSpecialRelaSection = m_writer.addRelocationSection();
    m_dmaTasksSpecialRelaSection->setName(".rela.dmaTasks_special");
    m_dmaTasksSpecialRelaSection->setSpecialSymbolTable(VPU_RT_SYMTAB);
    m_dmaTasksSpecialRelaSection->setSectionToPatch(m_dmaTasksSection);

}


void VPUIP::ManualELFBlobSerializer::addCmxDMA(uint8_t type){
    host_parsing::DmaDescriptor dma_descriptor; 
    host_parsing::DmaWrapper dma_wrapper;

    // set dma_config_bits
    host_parsing::DmaConfigBits dma_config_bits;
    dma_config_bits.type = 1;
    dma_config_bits.burst_length = 16;
    dma_config_bits.critical = 1;
    dma_config_bits.interrupt_en = 0;
    dma_config_bits.interrupt_trigger = 0;
    dma_config_bits.skip_nr = 0;
    dma_config_bits.order_forced = 0;
    dma_config_bits.watermark_en = 0;
    dma_config_bits.dec_en = 0;
    dma_config_bits.barrier_en = 1;
    // dma_config_bits.reserved

    // set dma Attr2D
    host_parsing::Dma2DAttributes dma_2d_attr;
    dma_2d_attr.src_stride = 256 * 256 * sizeof(uint32_t);
    dma_2d_attr.dst_stride = 256 * 256 * sizeof(uint32_t);
    dma_2d_attr.src_width = 256 * 256 * sizeof(uint32_t);
    dma_2d_attr.dst_width = 256 * 256 * sizeof(uint32_t);

    // set barriers
    host_parsing::DmaBarrierCfg dma_barrier_config;
    dma_barrier_config.cons_mask = 0;
    dma_barrier_config.prod_mask = 0;


    // set DmaDescriptor
    // dma_descriptor.link_address = 0;
    // dma_descriptor.reserved
    dma_descriptor.watermark = 0;
    dma_descriptor.cfg_link.cfg_bits = dma_config_bits;
    // uint64_t src;
    // uint64_t dst;
    dma_descriptor.length = 256 * 256 * sizeof(uint32_t);
    dma_descriptor.num_planes = 0;
    dma_descriptor.task_id = 0;
    dma_descriptor.src_plane_stride = 0;
    dma_descriptor.dst_plane_stride = 0;
    dma_descriptor.attr2d = dma_2d_attr;
    dma_descriptor.barriers = dma_barrier_config;
    dma_wrapper.transaction = dma_descriptor;

   
    if (type == DMA_INPUT){
        dma_wrapper.transaction.link_address = 1;
        dma_wrapper.start_after = 0;
        dma_wrapper.transaction.barriers.cons_mask = 0b0000000000000000;
        dma_wrapper.transaction.barriers.prod_mask = 0b0000000000000001;
    }
    if (type == DMA_OUTPUT){
        dma_wrapper.transaction.link_address = 0;
        dma_wrapper.start_after = 0;
        dma_wrapper.transaction.barriers.cons_mask = 0b0000000000000010;
        dma_wrapper.transaction.barriers.prod_mask = 0b0000000000000000;
    }
    

    m_dmaTasksSection->appendData(&dma_wrapper, 1);

    if (type == DMA_INPUT){
        m_networkInputSymbols = m_writer.addSymbolSection();
        m_networkInputSymbols->setName("inputs");
        m_networkInputSymbols->maskFlags(VPU_SHF_USERINPUT);

        m_inputSym = m_networkInputSymbols->addSymbolEntry();
        m_inputSym->setName("input");  // TODO: get name of tensor?
        m_inputSym->setType(VPU_STT_INPUT);
        m_inputSym->setValue(0);
        m_inputSym->setSize(IO_WIDTH * IO_HEIGHT * IO_CHANNELS * sizeof(IO_TYPE));
        
        auto inputRela = m_writer.addRelocationSection();
        inputRela->setName(".rela.dmaTasks_input");
        inputRela->setSymbolTable(m_networkInputSymbols);
        inputRela->setSectionToPatch(m_dmaTasksSection);
        inputRela->maskFlags(VPU_SHF_JIT);
        inputRela->maskFlags(VPU_SHF_USERINPUT);;

        auto src_reloc = inputRela->addRelocationEntry();
        src_reloc->setSymbol(m_inputSym);
        src_reloc->setType(R_VPU_64);
        src_reloc->setOffset(offsetof(host_parsing::DmaWrapper, transaction) +
                            offsetof(host_parsing::DmaDescriptor, src));
        src_reloc->setAddend(0);

        auto dst_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        dst_reloc->setSpecialSymbol(NNRD_SYM_NNCXM_SLICE_BASE_ADDR);
        dst_reloc->setType(R_VPU_64);
        dst_reloc->setOffset(offsetof(host_parsing::DmaWrapper, transaction) +
                            offsetof(host_parsing::DmaDescriptor, dst));
        dst_reloc->setAddend(0);

        auto prod_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        prod_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
        prod_mask_reloc->setType(R_VPU_64_LSHIFT);
        prod_mask_reloc->setOffset(offsetof(host_parsing::DmaWrapper, transaction) +
                                   offsetof(host_parsing::DmaDescriptor, barriers) +
                                   offsetof(host_parsing::DmaBarrierCfg, prod_mask));
        prod_mask_reloc->setAddend(0);

        auto cons_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        cons_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
        cons_mask_reloc->setType(R_VPU_64_LSHIFT);
        cons_mask_reloc->setOffset(offsetof(host_parsing::DmaWrapper, transaction) +
                                   offsetof(host_parsing::DmaDescriptor, barriers) +
                                   offsetof(host_parsing::DmaBarrierCfg, cons_mask));
        cons_mask_reloc->setAddend(0);

        auto link_address_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        link_address_reloc->setSpecialSymbol(NNRD_SYM_RTM_DMA0);
        link_address_reloc->setType(R_VPU_64_OR_RTM);
        link_address_reloc->setOffset(offsetof(host_parsing::DmaWrapper, transaction));
        link_address_reloc->setAddend(sizeof(host_parsing::DmaWrapper));

    }
    else if (type == DMA_OUTPUT){
        m_networkOutputSymbols = m_writer.addSymbolSection();
        m_networkOutputSymbols->setName("outputs");
        m_networkOutputSymbols->maskFlags(VPU_SHF_USEROUTPUT);

        m_outputSym = m_networkOutputSymbols->addSymbolEntry();
        m_outputSym->setName("output");
        m_outputSym->setType(VPU_STT_OUTPUT);
        m_outputSym->setValue(0);
        m_outputSym->setSize(IO_WIDTH * IO_HEIGHT * IO_CHANNELS * sizeof(IO_TYPE));

        auto outputRela = m_writer.addRelocationSection();
        outputRela->setName(".rela.dmaTasks_output");
        outputRela->setSymbolTable(m_networkOutputSymbols);
        outputRela->setSectionToPatch(m_dmaTasksSection);
        outputRela->maskFlags(VPU_SHF_JIT);
        outputRela->maskFlags(VPU_SHF_USEROUTPUT);;

        auto src_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        src_reloc->setSpecialSymbol(NNRD_SYM_NNCXM_SLICE_BASE_ADDR);
        src_reloc->setType(R_VPU_64);
        src_reloc->setOffset(sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction) +
                            offsetof(host_parsing::DmaDescriptor, src));
        src_reloc->setAddend(1000);

        auto dst_reloc = outputRela->addRelocationEntry();
        dst_reloc->setSymbol(m_outputSym);
        dst_reloc->setType(R_VPU_64);
        dst_reloc->setOffset(sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction) +
                             offsetof(host_parsing::DmaDescriptor, dst));
        dst_reloc->setAddend(0);

        auto prod_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        prod_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
        prod_mask_reloc->setType(R_VPU_64_LSHIFT);
        prod_mask_reloc->setOffset(sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction) +
                                   offsetof(host_parsing::DmaDescriptor, barriers) +
                                   offsetof(host_parsing::DmaBarrierCfg, prod_mask));
        prod_mask_reloc->setAddend(0);

        auto cons_mask_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        cons_mask_reloc->setSpecialSymbol(NNRD_SYM_BARRIERS_START);
        cons_mask_reloc->setType(R_VPU_64_LSHIFT);
        cons_mask_reloc->setOffset(sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction) +
                                   offsetof(host_parsing::DmaDescriptor, barriers) +
                                   offsetof(host_parsing::DmaBarrierCfg, cons_mask));
        cons_mask_reloc->setAddend(0);

        auto link_address_reloc = m_dmaTasksSpecialRelaSection->addRelocationEntry();
        link_address_reloc->setSpecialSymbol(NNRD_SYM_RTM_DMA0);
        link_address_reloc->setType(R_VPU_64_OR_RTM);
        link_address_reloc->setOffset(sizeof(host_parsing::DmaWrapper) + offsetof(host_parsing::DmaWrapper, transaction));
        link_address_reloc->setAddend(sizeof(host_parsing::DmaWrapper));
    }
}

host_parsing::DmaDescriptor VPUIP::ManualELFBlobSerializer::createEmptyDmaDescriptor(){

    host_parsing::DmaDescriptor dma_descriptor;

    host_parsing::DmaConfigBits dma_config_bits;
    dma_config_bits.type = 0;
    dma_config_bits.burst_length = 0;
    dma_config_bits.critical = 0;
    dma_config_bits.interrupt_en = 0;
    dma_config_bits.interrupt_trigger = 0;
    dma_config_bits.skip_nr = 0;
    dma_config_bits.order_forced = 0;
    dma_config_bits.watermark_en = 0;
    dma_config_bits.dec_en = 0;
    dma_config_bits.barrier_en = 0;
    dma_config_bits.reserved = 0;

    host_parsing::Dma2DAttributes dma_2d_attr;
    dma_2d_attr.src_stride = 0;
    dma_2d_attr.dst_stride = 0;
    dma_2d_attr.src_width = 0;
    dma_2d_attr.dst_width = 0;

    host_parsing::DmaBarrierCfg dma_barrier_config;
    dma_barrier_config.cons_mask = 0;
    dma_barrier_config.prod_mask = 0;

    dma_descriptor.link_address = 0;
    dma_descriptor.reserved = 0;
    dma_descriptor.watermark = 0;
    dma_descriptor.cfg_link.cfg_bits = dma_config_bits;
    dma_descriptor.src = 0;
    dma_descriptor.dst = 0;

    dma_descriptor.length = 0;
    dma_descriptor.num_planes = 0;
    dma_descriptor.task_id = 0;
    dma_descriptor.src_plane_stride = 0;
    dma_descriptor.dst_plane_stride = 0;
    dma_descriptor.attr2d = dma_2d_attr;
    dma_descriptor.barriers = dma_barrier_config;

    return dma_descriptor;

}



void VPUIP::ManualELFBlobSerializer::initBarriers() {
    m_barriersSection = m_writer.addBinaryDataSection<host_parsing::BarrierWrapper>();
    m_barriersSection->setName(".text.Barriers");
    m_barriersSection->setFlags(SHF_EXECINSTR);
    m_barriersSection->setAddrAlign(64);

    m_barriersSymbol = m_sectionSymbols->addSymbolEntry();
    m_barriersSymbol->setName("barriers");
    m_barriersSymbol->setRelatedSection(m_barriersSection);
    m_barriersSymbol->setType(STT_SECTION);
    m_sectionSymbolsMapping.insert(std::make_pair(m_barriersSection, m_barriersSymbol));
}

void VPUIP::ManualELFBlobSerializer::addLinearlyDependentBarrier(uint8_t type){
    
    host_parsing::BarrierWrapper barrier_wrapper;

    barrier_wrapper.producer_count = 1;
    barrier_wrapper.consumer_count = 1;

    int8_t barrier_num = -1;

    if (type == DMA_INPUT){
        barrier_num = 0;
    } 
    else if (type == DMA_OUTPUT){
        barrier_num = 1;
    }

    barrier_wrapper.next_same_id = -1;
    barrier_wrapper.real_id = barrier_num % 64;
    
    m_barriersSection->appendData(&barrier_wrapper, 1);
    m_barriersSymbol->setSize(m_barriersSection->getDataSize());

}

//*************************************

std::vector<char> VPUIP::ManualELFBlobSerializer::getBlob() {
    finalize();
    return m_writer.generateELF();
}

void VPUIP::ManualELFBlobSerializer::finalize() {
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

    // setup counts and fields that are used in MappedInference
    m_mappedInference.dmaTasks[0].count = 2;
    m_mappedInference.dmaTasks[1].count = 0;
    m_mappedInference.barrierConfigs.count = 2;


    // set unused fields to 0
    host_parsing::DmaDescriptor empty_dma_descriptor = createEmptyDmaDescriptor();

    m_mappedInference.feederDescriptors[0] = empty_dma_descriptor;
    m_mappedInference.feederDescriptors[1] = empty_dma_descriptor;
    m_mappedInference.feederDescriptors[2] = empty_dma_descriptor;
    m_mappedInference.feederDescriptors[3] = empty_dma_descriptor;
    m_mappedInference.feederDescriptors[4] = empty_dma_descriptor;

    m_mappedInference.leadingDmaCount[0] = 0;
    m_mappedInference.leadingDmaCount[1] = 0;
    m_mappedInference.variants.count = 0;
    m_mappedInference.invariants.count = 0;


    // add actRtConfigs to m_mappedInference before serialization
    m_mappedInference.actRtConfigs = setActRtConfigs();
    m_mappedInference.actKRanges.count = m_kernelsNum;
    m_mappedInference.actKInvocations.count = m_actKernelInvocations->getDataSize() / sizeof(host_parsing::ActKernelInvocationWrapper);
    

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

    if (m_dmaTasksSection) {

        auto mappedInferenceActKernelRangeRelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceActKernelRangeRelocation->setSymbol(m_dmaTasksSymbol);
        mappedInferenceActKernelRangeRelocation->setType(R_VPU_64);
        mappedInferenceActKernelRangeRelocation->setOffset(offsetof(host_parsing::MappedInference, dmaTasks) +
                                                offsetof(TaskReference<DmaWrapper>, address));
        mappedInferenceActKernelRangeRelocation->setAddend(0);
        segment->addSection(m_dmaTasksSection);
    }

    //
    // BarrierConfigs
    //

    if (m_barriersSection) {

        auto mappedInferenceActKernelRangeRelocation = mappedInferenceRelaSection->addRelocationEntry();
        mappedInferenceActKernelRangeRelocation->setSymbol(m_barriersSymbol);
        mappedInferenceActKernelRangeRelocation->setType(R_VPU_64);
        mappedInferenceActKernelRangeRelocation->setOffset(offsetof(host_parsing::MappedInference, barrierConfigs) +
                                                offsetof(TaskReference<BarrierWrapper>, address));
        mappedInferenceActKernelRangeRelocation->setAddend(0);
        segment->addSection(m_barriersSection);
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
        mappedInferenceActKernelRangeRelocation->setType(R_VPU_32);
        mappedInferenceActKernelRangeRelocation->setOffset(offsetof(host_parsing::MappedInference, actKRanges) +
                                                offsetof(TaskReference<ActKernelRangeWrapper>, address));
        mappedInferenceActKernelRangeRelocation->setAddend(0);
        segment->addSection(m_actKernelRanges);


        auto mappedInferenceActKernelInvocationRelocation = mappedInferenceActKernelsRelaSection->addRelocationEntry();
        mappedInferenceActKernelInvocationRelocation->setSymbol(m_actKernelInvocationWrapperSymbol);
        mappedInferenceActKernelInvocationRelocation->setType(R_VPU_32);
        mappedInferenceActKernelInvocationRelocation->setOffset(offsetof(host_parsing::MappedInference, actKInvocations) +
                                                offsetof(TaskReference<ActKernelInvocationWrapper>, address));
        mappedInferenceActKernelInvocationRelocation->setAddend(0);
        segment->addSection(m_actKernelInvocations);

        auto mappedInferenceActRtConfigs_main_Relocation = mappedInferenceActKernelsRelaSection->addRelocationEntry();
        mappedInferenceActRtConfigs_main_Relocation->setSymbol(m_actRtConfigMainSymbol);
        mappedInferenceActRtConfigs_main_Relocation->setType(R_VPU_32);
        mappedInferenceActRtConfigs_main_Relocation->setOffset(offsetof(host_parsing::MappedInference, actRtConfigs) +
                                                offsetof(host_parsing::ActKernelRuntimeConfigs, actRtWindowBase_));
        mappedInferenceActRtConfigs_main_Relocation->setAddend(0);

    }

}
