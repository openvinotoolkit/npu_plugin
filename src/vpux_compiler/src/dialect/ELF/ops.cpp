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

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include <elf/writer.hpp>
#include "llvm/Support/Debug.h"

//
// initialize
//

void vpux::ELF::ELFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/ELF/generated/types.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>

/*
void vpux::ELF::CreateSectionOp::serialize(elf::Writer& writer) {
}
*/

#define NUM_SECTIONS_MAX 100
#define NUM_SYMBOLS_MAX 10000
#define STR_ELF_SYMBOL "ELF.Symbol "

int ELFSectionIndex = 0;  // TODO: put somewhere else
struct ELFSectionAttributes {
    std::string sectionName;
    int sectionType;
    int sectionFlags;
    int sectionInfo;
    int sectionAddrAlignInfo;
    std::vector<char> serializedData;
} sectionAttributes[NUM_SECTIONS_MAX];
// elf::writer::BinaryDataSection<char>* ELFSection[NUM_SECTIONS_MAX];
elf::writer::Section* ELFSection[NUM_SECTIONS_MAX];
// elf::writer::SymbolSection ELFSection2[NUM_SECTIONS_MAX];

elf::writer::Symbol* ELFSymbol[NUM_SYMBOLS_MAX];
int ELFSymbolIndex = 0;  // TODO: put somewhere else
//
mlir::Value ELFSymbolValue[NUM_SYMBOLS_MAX];

vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

void vpux::ELF::PutAnyOpInSectionOp::serialize(
        /* elf::Writer& elfWriter */ std::vector<elf::writer::Section*> ELFSection, int ELFSectionIndex,
        std::vector<elf::writer::Symbol*> ELFSymbol, int ELFSymbolIndex) {
    // (void)elfWriter;
    (void)ELFSection;
    (void)ELFSectionIndex;
    (void)ELFSymbol;
    (void)ELFSymbolIndex;
}

void vpux::ELF::RelocOp::serialize(/*elf::Writer& elfWriter, */ std::vector<elf::writer::Section*> ELFSection,
                                   int ELFSectionIndex, std::vector<elf::writer::Symbol*> ELFSymbol,
                                   int ELFSymbolIndex) {
    // (void)elfWriter;
    (void)ELFSection;
    (void)ELFSectionIndex;
    (void)ELFSymbol;
    (void)ELFSymbolIndex;
    /*
        llvm::dbgs() << "Entered ELF::RelocOp::serialize()\n";
        llvm::dbgs().flush();

        // llvm::dbgs() << "processBlock(): Found ELF.RelocOp\n";
        // llvm::dbgs().flush();

        // vpux::ELF::RelocOp opReloc = llvm::cast<vpux::ELF::RelocOp>(op);

        llvm::dbgs() << "ELF::RelocOp::serialize(): offsetTargetField() = " << offsetTargetField() << "\n";

        // llvm::dbgs() << "processBlock(): relocationType() = " << relocationType().str() << "\n";
        llvm::dbgs() << "ELF::RelocOp::serialize(): relocationType() = " << static_cast<uint32_t>(relocationType()) <<
       "\n";

        llvm::dbgs() << "ELF::RelocOp::serialize(): sourceSymbol() = " << sourceSymbol() << "\n";
        llvm::dbgs() << "ELF::RelocOp::serialize(): addend() = " << addend() << "\n";

        int idx;
        for (idx = 0; idx < ELFSymbolIndex; idx++) {
            if (ELFSymbolValue[idx] == sourceSymbol())
                break;
        }
        llvm::dbgs() << "ELF::RelocOp::serialize(): Found idx = " << idx << "\n";

        auto relocationEntry = ((elf::writer::RelocationSection*)ELFSection[ELFSectionIndex])->addRelocationEntry();
        relocationEntry->setOffset(offsetTargetField());
        relocationEntry->setSymbol(ELFSymbol[idx]);  // ELFSymbol[ELFSymbolIndex]);
        relocationEntry->setAddend(addend());
    */
}

void vpux::ELF::CreateSymbolTableSectionOp::serialize(
        /* elf::Writer& elfWriter */ std::vector<elf::writer::Section*> ELFSection, int ELFSectionIndex,
        std::vector<elf::writer::Symbol*> ELFSymbol, int ELFSymbolIndex) {
    // (void)elfWriter;
    (void)ELFSection;
    (void)ELFSectionIndex;
    (void)ELFSymbol;
    (void)ELFSymbolIndex;

    llvm::dbgs() << "Entered ELF::CreateSymbolTableSectionOp::serialize()\n";
    llvm::dbgs().flush();

    /*
    sec = writer.createSection();
    std::vector<char> fullBinary;
    for (op in ops) { // ops sunt operaţiile efective din CreateSection
                  std::vector<char> binOp = op.serialize(); // care va chema funcţia de serializare a fiecărei operaţie
    în sine care face această structură de HglCmxDmaTransaction std::move(fullBinary.back(), binOp); // sau poate
    optimizăm şi dăm referinţe să nu tot alocăm şi copiem - optimizări mai târziu
    }
    sec.setData(fullBinary);
    sec.setSectionSize(fullBinary.size);
    // ...
    */
}
