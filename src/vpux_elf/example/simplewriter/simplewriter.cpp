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

    auto weights = elf.addSegment();
    weights->setType(elf::PT_LOAD);
    weights->addData("11111", 5);

    //
    // MappedInference data
    //

    auto mappedInferenceStruct = elf.addBinaryDataSection<MappedInference>();
    mappedInferenceStruct->setName(".text.MappedInference");
    mappedInferenceStruct->setAddrAlign(64);
    mappedInferenceStruct->addData(MappedInference());

    auto dmaTasks = elf.addBinaryDataSection<DMATask>();
    dmaTasks->setName(".text.dmaTasks");
    dmaTasks->setAddrAlign(64);
    dmaTasks->addData(DMATask());

    auto invariants = elf.addBinaryDataSection<Invariant>();
    invariants->setName(".text.invariants");
    invariants->setAddrAlign(64);
    invariants->addData(Invariant());

    auto variants = elf.addBinaryDataSection<Variant>();
    variants->setName(".text.variants");
    variants->setAddrAlign(64);
    variants->addData(Variant());
    variants->addData(Variant());
    variants->addData(Variant());

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

    auto outputDMA = dmaTasksRelocation->addRelocationEntry();
    outputDMA->setSymbol(output);
    outputDMA->setAddend(0);

    elf.write("nn.elf");

    return 0;
}
