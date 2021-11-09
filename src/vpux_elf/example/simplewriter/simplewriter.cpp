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

#include <elf/writer.hpp>

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

    auto symbolSection = elf.addSymbolSection();
    symbolSection->setName(".symtab");

    auto input = symbolSection->addSymbolEntry();
    input->setName(".input");

    auto output = symbolSection->addSymbolEntry();
    output->setName(".output");

    //
    // Weights
    //

    std::vector<uint64_t> weightsData{0, 1, 2, 3, 4, 5};
    auto weightsSection = elf.addBinaryDataSection<uint64_t>();
    weightsSection->setName(".weights");
    weightsSection->appendData(weightsData.data(), weightsData.size());

    auto weightsSegment = elf.addSegment();
    weightsSegment->setType(elf::PT_LOAD);
    weightsSegment->addSection(weightsSection);

    auto weightsSym = symbolSection->addSymbolEntry();
    weightsSym->setRelatedSection(weightsSection);

    //
    // MappedInference data
    //

    auto mappedInferenceStruct = elf.addBinaryDataSection<MappedInference>();
    mappedInferenceStruct->setName(".text.MappedInference");
    mappedInferenceStruct->setAddrAlign(64);
    mappedInferenceStruct->appendData(MappedInference());

    auto dmaTasks = elf.addBinaryDataSection<DMATask>();
    dmaTasks->setName(".text.dmaTasks");
    dmaTasks->setAddrAlign(64);
    dmaTasks->appendData(DMATask());
    dmaTasks->appendData(DMATask());
    dmaTasks->appendData(DMATask());
    dmaTasks->appendData(DMATask());

    auto invariants = elf.addBinaryDataSection<Invariant>();
    invariants->setName(".text.invariants");
    invariants->setAddrAlign(64);
    invariants->appendData(Invariant());

    auto variants = elf.addBinaryDataSection<Variant>();
    variants->setName(".text.variants");
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

    auto dmaTasksRelocation = elf.addRelocationSection();
    dmaTasksRelocation->setSymbolTable(symbolSection);
    dmaTasksRelocation->setSectionToPatch(dmaTasks);
    dmaTasksRelocation->setName(".rela.dma");

    auto inputDMA = dmaTasksRelocation->addRelocationEntry();
    inputDMA->setSymbol(input);
    inputDMA->setAddend(0);
    inputDMA->setOffset(0);

    auto weightsDMA = dmaTasksRelocation->addRelocationEntry();
    weightsDMA->setSymbol(weightsSym);
    weightsDMA->setAddend(0);
    weightsDMA->setOffset(8);

    auto outputDMA = dmaTasksRelocation->addRelocationEntry();
    outputDMA->setSymbol(output);
    outputDMA->setAddend(0);
    outputDMA->setOffset(4);

    const auto elfBlob = elf.generateELF();

    std::ofstream stream("test.elf", std::ios::out | std::ios::binary);
    stream.write(elfBlob.data(), elfBlob.size());

    return 0;
}
