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

void VPUIP::ELFBlobSerializer::setNetworkInputs(llvm::ArrayRef<mlir::ShapedType> inputs) {
    setNetworkIO(inputs, VPU_STT_INPUT, m_networkInputSymbols, "input");
}

void VPUIP::ELFBlobSerializer::setNetworkOutputs(llvm::ArrayRef<mlir::ShapedType> outputs) {
    setNetworkIO(outputs, VPU_STT_OUTPUT, m_networkOutputSymbols, "output");
}

void VPUIP::ELFBlobSerializer::setLeadingDMACount(uint32_t leadingDMACount, size_t dmaEngineIndex) {
    m_mappedInference.leadingDmaCount[dmaEngineIndex] = leadingDMACount;
}

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

void VPUIP::ELFBlobSerializer::setBarrierConfigs(llvm::ArrayRef<BarrierWrapper> barrierConfigs) {
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

void VPUIP::ELFBlobSerializer::setWeights(llvm::ArrayRef<llvm::ArrayRef<uint64_t>> weights) {
    m_weightsOffsets.reserve(weights.size());
    m_weightsOffsets.push_back(0);
    for (size_t i = 1; i < weights.size(); ++i) {
        m_weightsOffsets.push_back(m_weightsOffsets[i - 1] + weights[i - 1].size());
    }

    std::vector<uint64_t> weightsVec;
    for (const auto& weightTensor : weights) {
        weightsVec.insert(weightsVec.end(), weightTensor.begin(), weightTensor.end());
    }

    m_weights = m_writer.addBinaryDataSection<uint64_t>();
    m_weights->appendData(weightsVec.data(), weightsVec.size());
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
    // MappedInference
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

    finalizeDMA();
    for (size_t i = 0; i < m_dmaTasksSections.size(); ++i) {
        const auto& dmaTasks = m_dmaTasksSections[i];
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
    // DPUTasks
    //

    finalizeDPU();
    if (m_dpuInvariants) {
        VPUX_THROW_UNLESS(m_dpuVariants, "DPU variants can't be NULL");

        auto invariantsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        invariantsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_dpuInvariants));
        invariantsRelocation->setType(R_VPU_64);
        invariantsRelocation->setOffset(offsetof(MappedInference, invariants) +
                                        offsetof(TaskReference<DPUInvariantWrapper>, address));
        invariantsRelocation->setAddend(0);
        segment->addSection(m_dpuInvariants);

        auto variantsRelocation = mappedInferenceRelaSection->addRelocationEntry();
        variantsRelocation->setSymbol(m_sectionSymbolsMapping.at(m_dpuVariants));
        variantsRelocation->setType(R_VPU_64);
        variantsRelocation->setOffset(offsetof(MappedInference, variants) +
                                      offsetof(TaskReference<DPUVariantWrapper>, address));
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
        mappedInferenceBarrierConfigsRelocation->setOffset(offsetof(MappedInference, barrierConfigs) +
                                                           offsetof(TaskReference<DmaWrapper>, address));
        mappedInferenceBarrierConfigsRelocation->setAddend(0);
        segment->addSection(m_barrierConfigs);
    }
}

void VPUIP::ELFBlobSerializer::finalizeDMA() {
    for (size_t dmaEngineIndex = 0; dmaEngineIndex < m_dmaTasks.size(); ++dmaEngineIndex) {
        auto& dmaTasks = m_dmaTasks[dmaEngineIndex];
        if (dmaTasks.empty()) {
            continue;
        }

        auto& dmaTasksSection = m_dmaTasksSections[dmaEngineIndex];
        dmaTasksSection = m_writer.addBinaryDataSection<DmaWrapper>();
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

            const auto transactionOffset = i * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction);
            relocationManager.addRelocation(dmaTask.input, R_VPU_64, transactionOffset + offsetof(DmaDescriptor, src));
            relocationManager.addRelocation(dmaTask.output, R_VPU_64, transactionOffset + offsetof(DmaDescriptor, dst));

            if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::DDR_DMA) {
                relocationManager.addRelocation(
                        m_sectionSymbols, dmaTasksSymbol, R_VPU_64_OR,
                        dmaTask.linkAddress.dmaTaskIndex * sizeof(DmaWrapper) + offsetof(DmaWrapper, transaction),
                        transactionOffset);
            } else if (dmaTask.linkAddress.metaDataLocation == LinkAddressPatchingInfo::MetaDataLocation::RTM_DMA) {
                dmaTask.dmaDescriptor.transaction.link_address = dmaTask.linkAddress.dmaTaskIndex;
                relocationManager.addRelocation(NNRD_SYM_RTM_DMA0 + dmaEngineIndex, R_VPU_64_OR_RTM, sizeof(DmaWrapper),
                                                transactionOffset);
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

        std::vector<DmaWrapper> dmaDescriptors;
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

    m_dpuInvariants = m_writer.addBinaryDataSection<DPUInvariantWrapper>();
    m_dpuInvariants->setName(".text.Invariants");
    m_dpuInvariants->setFlags(SHF_EXECINSTR);
    m_dpuInvariants->setAddrAlign(64);

    auto invariantsSymbol = m_sectionSymbols->addSymbolEntry();
    invariantsSymbol->setName(".ddr.Invariants");
    invariantsSymbol->setRelatedSection(m_dpuInvariants);
    invariantsSymbol->setType(STT_SECTION);
    m_sectionSymbolsMapping.insert(std::make_pair(m_dpuInvariants, invariantsSymbol));

    m_dpuVariants = m_writer.addBinaryDataSection<DPUVariantWrapper>();
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

        const auto reduceWaitMaskTo8bit = [](BarrierConfig& barrierConfig) {
            barrierConfig.group = 0;
            barrierConfig.mask = 0;

            for (uint64_t mask = barrierConfig.consumer_mask, group = 1; mask > 0; mask >>= 8, ++group) {
                if (mask & 0xff) {
                    // Store 8-bit barrier group (1-based) and mask to be used by ShaveNN

                    if (barrierConfig.group == 0) {
                        barrierConfig.group = static_cast<unsigned char>(group);
                        barrierConfig.mask = mask & 0xff;

                        // Keep checking the wait_mask_ for a second match
                    } else {
                        // When multiple groups match, GPIO cannot be used
                        barrierConfig.group = 0;
                        barrierConfig.mask = 0;
                        return;
                    }
                }
            }
        };

        // TODO: works only for 0-th tile
        reduceWaitMaskTo8bit(dpuTask.dpuInvariant.dpuInvariantWrapper.invariant.barriers);

        for (auto& variant : m_dpuTasks[i].dpuVariants) {
            variant.dpuVariantWrapper.variant.invariant_addr = variant.dpuVariantWrapper.invariant_index;
            variantRelocationManager.addRelocation(NNRD_SYM_RTM_IVAR, R_VPU_32_OR_RTM, sizeof(DPUInvariantWrapper),
                                                   variantIndex * sizeof(DPUVariantWrapper) +
                                                           offsetof(DPUVariantWrapper, variant) +
                                                           offsetof(DPUVariant, invariant_addr));
            variantRelocationManager.addRelocation(m_dpuTasks[i].dpuInvariant.weightsTable, R_VPU_32_SUM,
                                                   variantIndex * sizeof(DPUVariantWrapper) +
                                                           offsetof(DPUVariantWrapper, variant) +
                                                           offsetof(DPUVariant, weight_table_offset));
            variant.dpuVariantWrapper.invariant_index = i;
            variantIndex++;
        }
    }

    std::vector<DPUInvariantWrapper> dpuInvariantWrappers;
    std::vector<DPUVariantWrapper> dpuVariantWrappers;
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
                                        invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                offsetof(DPUInvariantRegisters, act_offset) +
                                                i * sizeof(DPUInvariantRegisters::act_offset[0]));
    }

    auto& invariant = invariantTask.dpuInvariantWrapper.invariant;

    invariant.registers.se_sp_addr[1].se_addr = ((1 * SLICE_LENGTH) >> 4);
    invariant.registers.se_sp_addr[2].se_addr = ((2 * SLICE_LENGTH) >> 4);
    invariant.registers.se_sp_addr[3].se_addr = ((3 * SLICE_LENGTH) >> 4);

    // FIXME: hardcoded and directly copied from POC runtime...
    invariant.registers.base_offset_a = 0x200;
    invariant.registers.base_offset_b = 0x602;

    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
        const auto seSpAddr = invariantSectionOffset + offsetof(DPUInvariant, registers) +
                              offsetof(DPUInvariantRegisters, se_sp_addr) +
                              sizeof(DPUInvariantRegisters::se_sp_addr[0]);

        relocationManager.addRelocation(input, R_VPU_32, seSpAddr, OffsetToUse::SPARSITY_TABLE);
        relocationManager.addRelocation(input, R_VPU_32,
                                        seSpAddr + sizeof(DPUInvariantRegisters::se_sp_addr[0].se_addr),
                                        OffsetToUse::SPARSITY_MAP);
    }

    for (size_t i = 0; i < 4; ++i) {
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_BASE_SUB,
                                        invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                offsetof(DPUInvariantRegisters, base_adr) +
                                                i * sizeof(DPUInvariantRegisters::base_adr));
    }

    for (unsigned int i = 0; i < numSlices; ++i) {
        invariant.registers.odu_cast[i].odu_cast_bf.cast_enable = i;
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_OFFSET_CMP_OR,
                                        invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                offsetof(DPUInvariantRegisters, odu_cast) +
                                                i * sizeof(DPUInvariantRegisters::odu_cast[0]));

        invariant.registers.odu_cast[i].odu_cast_bf.cast_offset = i;
        relocationManager.addRelocation(invariantTask.output, R_VPU_32_MULTICAST_OFFSET_4_BIT_SHIFT_OR,
                                        invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                offsetof(DPUInvariantRegisters, odu_cast) +
                                                i * sizeof(DPUInvariantRegisters::odu_cast[0]));
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_pt) {
        relocationManager.addRelocation(
                invariantTask.output, R_VPU_32_MULTICAST_BASE,
                invariantSectionOffset + offsetof(DPUInvariant, registers) + offsetof(DPUInvariantRegisters, pt_base),
                OffsetToUse::SPARSITY_TABLE);
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_sp) {
        relocationManager.addRelocation(
                invariantTask.output, R_VPU_32_MULTICAST_BASE,
                invariantSectionOffset + offsetof(DPUInvariant, registers) + offsetof(DPUInvariantRegisters, sp_base),
                OffsetToUse::SPARSITY_MAP);
    }

    relocationManager.addRelocation(NNRD_SYM_FIFO_BASE, R_VPU_32, invariantTask.dpuInvariantWrapper.cluster,
                                    invariantSectionOffset * offsetof(DPUInvariantWrapper, cluster));

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
                                        invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                offsetof(DPUInvariantRegisters, weight_start));

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
                const auto seSpAddr = invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                      offsetof(DPUInvariantRegisters, se_sp_addr) +
                                      (i + 1) * sizeof(DPUInvariantRegisters::se_sp_addr[0]);

                if (invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                    relocationManager.addRelocation(input, R_VPU_32_SUM, seSpAddr);
                } else {
                    // HW issue (A0): se_addr for segments 2+ need and offset from the real address of the segment.
                    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                        relocationManager.addRelocation(input, R_VPU_32, seSpAddr, OffsetToUse::SPARSITY_TABLE);
                    }

                    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
                        relocationManager.addRelocation(input, R_VPU_32,
                                                        seSpAddr + sizeof(DPUInvariantRegisters::se_sp_addr[0].se_addr),
                                                        OffsetToUse::SPARSITY_MAP);
                    }

                    // Previous layers have set the ODU base select to the cluster index
                    // Need to have matching logic at IDU side
                    relocationManager.addRelocation(input, R_VPU_32,
                                                    invariantSectionOffset + offsetof(DPUInvariant, registers) +
                                                            offsetof(DPUInvariantRegisters, act_offset) +
                                                            (i + 1) * sizeof(DPUInvariantRegisters::act_offset[0]));
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
                offset + m_elfBlobSerializer.m_weightsOffsets[tensorPatchingInfo.location.locationIndex],
                sectionOffset);
        break;
    default:
        VPUX_THROW("Unsupported MemoryLocation {}", tensorPatchingInfo.location.memLocation);
        break;
    }
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

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::getRelocationSection(
        const elf::writer::SymbolSection* symbolSection, VPUIP::MemoryLocation memoryLocation) {
    if (m_symbolTableToRelocation.find(symbolSection) == m_symbolTableToRelocation.end()) {
        auto relocationTable = createRelocationSection(symbolSection);
        if (memoryLocation == VPUIP::MemoryLocation::ProgrammableInput) {
            relocationTable->maskFlags(VPU_SHF_JIT);
            relocationTable->maskFlags(VPU_SHF_USERINPUT);
        } else if (memoryLocation == VPUIP::MemoryLocation::ProgrammableOutput) {
            relocationTable->maskFlags(VPU_SHF_JIT);
            relocationTable->maskFlags(VPU_SHF_USEROUTPUT);
        }
        m_symbolTableToRelocation[symbolSection] = relocationTable;
    }

    return m_symbolTableToRelocation.at(symbolSection);
}

writer::RelocationSection* VPUIP::ELFBlobSerializer::RelocationManager::createRelocationSection(
        const writer::SymbolSection* symbolSection) {
    auto relocationTable = m_elfBlobSerializer.m_writer.addRelocationSection();
    relocationTable->setName(m_relocationSectionName);
    relocationTable->setSectionToPatch(m_sectionToPatch);
    relocationTable->setSymbolTable(symbolSection);
    return relocationTable;
}
