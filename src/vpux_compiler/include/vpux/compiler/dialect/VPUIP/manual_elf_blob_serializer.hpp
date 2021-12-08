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

#define IO_WIDTH 256
#define IO_HEIGHT 256
#define IO_CHANNELS 1
#define IO_TYPE uint32_t

#define DMA_INPUT 0
#define DMA_OUTPUT 1

namespace vpux {
namespace VPUIP {

enum class OffsetToUse { BASE = 0, DATA = 0, SPARSITY_MAP, SPARSITY_TABLE };

class ManualELFBlobSerializer {
public:
    ManualELFBlobSerializer();

    // Act Kernels
    void initActKernel(std::vector<char> elfBlob, std::string name);
    void addActKernel();
    void addActInvocation();
    host_parsing::ActKernelRuntimeConfigs setActRtConfigs();

    // DMA
    void initCmxDMA();
    void addCmxDMA(uint8_t type);

    // Barriers
    void initBarriers();
    void addLinearlyDependentBarrier(uint8_t type);

    std::vector<char> getBlob();

private:

    host_parsing::DmaDescriptor createEmptyDmaDescriptor();
    void finalize();

    elf::Writer m_writer;

    elf::writer::EmptySection* m_ddrScratch;
    host_parsing::ResourceRequirements m_resourceRequirements;
    host_parsing::MappedInference m_mappedInference;

    elf::writer::SymbolSection* m_sectionSymbols = nullptr;
    std::unordered_map<elf::writer::Section*, elf::writer::Symbol*> m_sectionSymbolsMapping;

    elf::writer::BinaryDataSection<host_parsing::DPUInvariantWrapper>* m_dpuInvariants = nullptr;
    elf::writer::BinaryDataSection<host_parsing::DPUVariantWrapper>* m_dpuVariants = nullptr;
    elf::writer::BinaryDataSection<uint8_t>* m_weights = nullptr;

    // ALL ABOUT DMA
    elf::writer::BinaryDataSection<host_parsing::DmaWrapper>* m_dmaTasksSection = nullptr;
    elf::writer::Symbol* m_dmaTasksSymbol = nullptr;
    elf::writer::SymbolSection* m_networkInputSymbols = nullptr;
    elf::writer::SymbolSection* m_networkOutputSymbols = nullptr;
    elf::writer::RelocationSection* m_dmaTasksSpecialRelaSection = nullptr;

    // ALL ABOUT BARRIERS
    elf::writer::BinaryDataSection<host_parsing::BarrierWrapper>* m_barriersSection = nullptr;
    elf::writer::Symbol* m_barriersSymbol = nullptr;


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
    elf::writer::BinaryDataSection<host_parsing::ActKernelInvocationWrapper>* m_actKernelInvocations = nullptr;
    elf::writer::RelocationSection* m_actKernelInvocationRela = nullptr;
    elf::writer::RelocationSection* m_actKernelInvocationSpecialRela = nullptr;

    // Act Kernel Rt Configs
    elf::writer::Symbol* m_actRtConfigMainSymbol = nullptr;
    elf::writer::Symbol* m_inputSym = nullptr;
    elf::writer::Symbol* m_outputSym = nullptr;

};

}  // namespace VPUIP
}  // namespace vpux
