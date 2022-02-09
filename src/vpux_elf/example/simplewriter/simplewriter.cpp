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

#include <vpux_elf/writer.hpp>

#include <fstream>

struct DMATask {
    int x;
    double y;
};

struct Invariant {
    float c;
};

struct Variant {
    char z;
};

struct Handle {
    uint32_t ptr;
    uint64_t size;
};

struct MappedInference {
    Handle dma;
    Handle inv;
    Handle var;
};

int main() {
    elf::Writer elf;

    //
    // SymbolSection
    //

    auto symbolSection = elf.addSymbolSection(".symtab");

    auto input = symbolSection->addSymbolEntry(".input");

    auto output = symbolSection->addSymbolEntry(".output");

    //
    // Weights
    //

    std::vector<uint64_t> weightsData{0, 1, 2, 3, 4, 5};
    auto weightsSection = elf.addBinaryDataSection<uint64_t>(".weights");
    weightsSection->appendData(weightsData.data(), weightsData.size());

    auto weightsSegment = elf.addSegment();
    weightsSegment->setType(elf::PT_LOAD);
    weightsSegment->addSection(weightsSection);

    auto weightsSym = symbolSection->addSymbolEntry(".weights");
    weightsSym->setRelatedSection(weightsSection);

    //
    // MappedInference data
    //

    auto mappedInferenceStruct = elf.addBinaryDataSection<MappedInference>(".text.MappedInference");
    mappedInferenceStruct->setAddrAlign(64);
    mappedInferenceStruct->appendData(MappedInference());

    auto dmaTasks = elf.addBinaryDataSection<DMATask>(".text.dmaTasks");
    dmaTasks->setAddrAlign(64);
    const auto inputDMAOffset = dmaTasks->appendData(DMATask());

    auto invariants = elf.addBinaryDataSection<Invariant>(".text.invariants");
    invariants->setAddrAlign(64);
    invariants->appendData(Invariant());

    auto variants = elf.addBinaryDataSection<Variant>(".text.variants");
    variants->setAddrAlign(64);
    variants->appendData(Variant());
    variants->appendData(Variant());
    variants->appendData(Variant());

    auto mappedInferenceSegment = elf.addSegment();
    mappedInferenceSegment->setType(elf::PT_LOAD);
    mappedInferenceSegment->addSection(mappedInferenceStruct);
    mappedInferenceSegment->addSection(dmaTasks);
    mappedInferenceSegment->addSection(invariants);
    mappedInferenceSegment->addSection(variants);

    //
    // Relocations
    //

    auto dmaTasksRelocation = elf.addRelocationSection(".rela.dma");
    dmaTasksRelocation->setSymbolTable(symbolSection);
    dmaTasksRelocation->setSectionToPatch(dmaTasks);

    auto inputDMA = dmaTasksRelocation->addRelocationEntry();
    inputDMA->setSymbol(input);
    inputDMA->setAddend(0);
    inputDMA->setOffset(inputDMAOffset + offsetof(DMATask, x));

    auto weightsDMA = dmaTasksRelocation->addRelocationEntry();
    weightsDMA->setSymbol(weightsSym);
    weightsDMA->setAddend(0);
    weightsDMA->setOffset(inputDMAOffset + offsetof(DMATask, x));

    auto outputDMA = dmaTasksRelocation->addRelocationEntry();
    outputDMA->setSymbol(output);
    outputDMA->setAddend(0);
    outputDMA->setOffset(inputDMAOffset + offsetof(DMATask, y));

    const auto elfBlob = elf.generateELF();

    std::ofstream stream("test.elf", std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(elfBlob.data()), elfBlob.size());

    return 0;
}
