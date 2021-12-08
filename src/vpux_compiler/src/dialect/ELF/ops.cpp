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

int ELFSectionIndex = 0;  // TODO: put somewhere else
#define NUM_SECTIONS_MAX 100
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

#define NUM_SYMBOLS_MAX 10000
elf::writer::Symbol* ELFSymbol[NUM_SYMBOLS_MAX];
int ELFSymbolIndex = 0;  // TODO: put somewhere else
//
mlir::Value ELFSymbolValue[NUM_SYMBOLS_MAX];

#define STR_ELF_SYMBOL "ELF.Symbol "

vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

void vpux::ELF::PutAnyOpInSectionOp::serialize(elf::Writer& elfWriter) {
    (void)elfWriter;

    std::vector<char> buffer;

    llvm::dbgs() << "Entered ELF::PutAnyOpInSectionOp::serialize()\n";
    llvm::dbgs().flush();

    // mlir::Value putAnyOpValue = llvm::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op).inputArg(); // barrier();
    mlir::Value putAnyOpValue = inputArg();  // barrier();

    llvm::dbgs() << "    putAnyOpValue = " << putAnyOpValue << "\n";
    // See https://llvm.org/doxygen/classllvm_1_1raw__ostream.html
    llvm::dbgs().flush();

    // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
    mlir::Operation* putAnyOpOp = putAnyOpValue.getDefiningOp();
    llvm::dbgs() << "    *putAnyOpOp = " << *putAnyOpOp << "\n";
    llvm::dbgs().flush();

    if (vpux::VPUIPRegMapped::ConfigureBarrierOp putAnyOpOpCfg =
                llvm::dyn_cast<vpux::VPUIPRegMapped::ConfigureBarrierOp>(putAnyOpOp)) {
        llvm::dbgs() << "    putAnyOpOpCfg is of type vpux::VPUIPRegMapped::ConfigureBarrierOp\n";
        llvm::dbgs() << "    putAnyOpOpCfg = " << putAnyOpOpCfg << "\n";
        llvm::dbgs().flush();
        putAnyOpOpCfg.serialize(buffer);

        llvm::dbgs() << "  Writing buffer for ConfigureBarrierOp to "
                        "sectionAttributes[ELFSectionIndex].serializedData, with ELFSectionIndex = "
                     << ELFSectionIndex << "\n";
        llvm::dbgs() << "    (buffer.size() = " << buffer.size() << ")\n";
        for (std::size_t i = 0; i < buffer.size(); i++) {
            // 2021_10_19: ConfigureBarriersELFBLOB.push_back(buffer[i]);
            sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);
            // sectionAttributes[ELFSectionIndex].serializedData.push_back(i);  // 2021_10_19
        }
    } else {
        llvm::dbgs() << "    putAnyOpOp is NOT of type vpux::VPUIPRegMapped::ConfigureBarrierOp\n";
        llvm::dbgs().flush();

        if (vpux::VPUIPRegMapped::DeclareBufferOp putAnyOpOpDB =
                    llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareBufferOp>(putAnyOpOp)) {
            llvm::dbgs() << "    putAnyOpOpDB is of type vpux::VPUIPRegMapped::DeclareBufferOp\n";
            llvm::dbgs() << "    putAnyOpOpDB = " << putAnyOpOpDB << "\n";
            llvm::dbgs().flush();
        } else {
            llvm::dbgs() << "    putAnyOpOp is NOT of type vpux::VPUIPRegMapped::DeclareBufferOp\n";
            llvm::dbgs().flush();

            if (vpux::Const::DeclareOp putAnyOpOpDcl = llvm::dyn_cast<vpux::Const::DeclareOp>(putAnyOpOp)) {
                llvm::dbgs() << "    putAnyOpOpDcl is of type vpux::Const::DeclareOp\n";
                llvm::dbgs() << "    putAnyOpOpDcl = " << putAnyOpOpDcl << "\n";
                llvm::dbgs().flush();

                // putAnyOpOpDcl.content();
                // Defined in include/vpux/compiler/dialect/const/utils/content.hpp
                // getValues()
                /*
                for (std::size_t i = 0; i < buffer.size(); i++) {
                    // sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                    sectionAttributes[ELFSectionIndex].serializedData.push_back(i);  // 2021_10_19
                }
                */

                putAnyOpOpDcl.serialize(buffer);

                llvm::dbgs() << "  Writing buffer for Const::DeclareOp to "
                                "sectionAttributes[ELFSectionIndex].serializedData, with ELFSectionIndex = "
                             << ELFSectionIndex << "\n";
                llvm::dbgs() << "    (buffer.size() = " << buffer.size() << ")\n";
                for (std::size_t i = 0; i < buffer.size(); i++) {
                    int16_t tmp16 = buffer[i] & 0xFF;
                    // printf("buffer[%lu] = 0x%hX\n", i, buffer[i]);
                    // Gives error: <<error: no match for ‘operator<<’...>> llvm::dbgs() << "    buffer[" << i
                    //   << "] = " << std::hex << tmp16 << "\n";
                    llvm::dbgs() << "    buffer[" << i << "] = 0x";
                    // See https://llvm.org/doxygen/classllvm_1_1raw__ostream.html
                    llvm::dbgs().write_hex(tmp16);
                    llvm::dbgs() << "\n";

                    // 2021_10_19: ConfigureBarriersELFBLOB.push_back(buffer[i]);
                    sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                    // sectionAttributes[ELFSectionIndex].serializedData.push_back(i + 128);  // 2021_10_19
                }
            } else {
                llvm::dbgs() << "    putAnyOpOp is NOT of type vpux::Const::DeclareOp\n";
                llvm::dbgs().flush();

                // IMPORTANT: We can have a memref or a Section
                if (vpux::ELF::SymbolOp putAnyOpOpES = llvm::dyn_cast<vpux::ELF::SymbolOp>(putAnyOpOp)) {
                    llvm::dbgs() << "    putAnyOpOpES is of type vpux::ELF::SymbolOp\n";
                    llvm::dbgs().flush();

                    llvm::dbgs() << "    putAnyOpOpES = " << putAnyOpOpES << "\n";

                    mlir::Value putAnyOpOpInputArg = putAnyOpOpES.inputArg();
                    // Basically it is equivalent to putAnyOpOpES:
                    mlir::Value putAnyOpOpSymbol = putAnyOpOpES.symbol();
                    //
                    llvm::dbgs() << "    putAnyOpOpInputArg = " << putAnyOpOpInputArg << "\n";
                    llvm::dbgs() << "    putAnyOpOpSymbol = " << putAnyOpOpSymbol << "\n";

                    // Note: This is normally a memref

                    // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
                    // mlir::Operation* putAnyOpOpInputArg_op = putAnyOpOpInputArg.getDefiningOp();
                    // llvm::dbgs() << "    *putAnyOpOpInputArg_op = " << *putAnyOpOpInputArg_op << "\n";
                    // llvm::dbgs().flush();

                    // auto input =
                    // ((elf::writer::SymbolSection*)ELFSection[ELFSectionIndex])->addSymbolEntry();
                    // input->setName(".input123"); // TODO: generate a new name each time
                    ELFSymbol[ELFSymbolIndex] =
                            ((elf::writer::SymbolSection*)ELFSection[ELFSectionIndex])->addSymbolEntry();

                    std::string printStr;
                    llvm::raw_string_ostream OS(printStr);
                    //
                    // std::string symbolNameStr = "Val";
                    std::string symbolNameStr;

                    mlir::BlockArgument blockArg = putAnyOpOpInputArg.dyn_cast_or_null<mlir::BlockArgument>();
                    if (blockArg) {
                        unsigned int blockArgNum = blockArg.getArgNumber();

                        llvm::dbgs() << "    blockArgNum = " << blockArgNum << "\n";
                        vpux::IE::DataInfoOp respectiveNetArg;
                        if (blockArgNum < diOpInVec.size()) {
                            respectiveNetArg = diOpInVec[blockArgNum];
                        } else {
                            respectiveNetArg = diOpOutVec[blockArgNum - diOpInVec.size()];
                        }
                        llvm::dbgs() << "    respectiveNetArg = " << respectiveNetArg << "\n";

                        // From https://llvm.org/doxygen/classllvm_1_1StringRef.html
                        symbolNameStr = respectiveNetArg.name().str();
                    } else {
                        // llvm::raw_ostream osStr;
                        // putAnyOpOpSymbol.print(OS);
                        putAnyOpOpES.print(OS);
                        llvm::dbgs() << "    printStr = " << printStr << "\n";

                        // We recover the number of the variable mlir::Value
                        // std::size_t posFound = printStr.find(' ');
                        std::size_t posFound = printStr.find(STR_ELF_SYMBOL);
                        if (posFound != std::string::npos) {
                            // symbolNameStr += printStr.substr(1, posFound);
                            // symbolNameStr = printStr.substr(posFound + strlen(STR_ELF_SYMBOL), printStr.size());
                            // Note: For inputArg() instead of e.g. %arg0 it appears something like "<block
                            // argument> of type ..."
                            symbolNameStr = printStr.substr(posFound + strlen(STR_ELF_SYMBOL) + 1, printStr.size());

                            std::size_t posFound2 = symbolNameStr.find(' ');
                            symbolNameStr = symbolNameStr.substr(0, posFound2);
                            // symbolNameStr = printStr.substr(posFound + strlen(STR_ELF_SYMBOL) + 1,
                            // printStr.size());
                        } else {
                            symbolNameStr = printStr;
                        }
                    }

                    llvm::dbgs() << "    symbolNameStr = " << symbolNameStr << "\n";

                    // ELFSymbol[ELFSymbolIndex]->setName(".input123");
                    ELFSymbol[ELFSymbolIndex]->setName(symbolNameStr);
                    ELFSymbolValue[ELFSymbolIndex] = putAnyOpOpSymbol;

                    ELFSymbol[ELFSymbolIndex]->setValue(0);

                    mlir::Type inputArgType = putAnyOpOpInputArg.getType();  // diOpInVec[idx].userType();
                    int64_t inputArgTypeTotalSize = -1;
                    if (mlir::ShapedType sType = inputArgType.dyn_cast<mlir::ShapedType>()) {
                        const Byte aByte = vpux::getTotalSize(sType);
                        // uint32_t tmp = static_cast<uint32_t>(tmpBit);
                        inputArgTypeTotalSize = aByte.count();
                        llvm::dbgs() << "  inputArgTypeTotalSize = " << inputArgTypeTotalSize << "\n";
                    }
                    ELFSymbol[ELFSymbolIndex]->setSize(inputArgTypeTotalSize);

                    llvm::dbgs() << "    ELFSymbolIndex = " << ELFSymbolIndex << "\n";
                    ELFSymbolIndex++;
                } else {
                    llvm::dbgs() << "    putAnyOpOpES is NOT of type vpux::ELF::SymbolOp\n";
                    llvm::dbgs().flush();
                }
            }
        }
    }
}

void vpux::ELF::RelocOp::serialize(elf::Writer& elfWriter) {
    (void)elfWriter;

    llvm::dbgs() << "Entered ELF::RelocOp::serialize()\n";
    llvm::dbgs().flush();

    llvm::dbgs() << "processBlock(): Found ELF.RelocOp\n";
    llvm::dbgs().flush();

    // vpux::ELF::RelocOp opReloc = llvm::cast<vpux::ELF::RelocOp>(op);

    /*
    llvm::dbgs() << "processBlock(): offsetTargetField() = " << relocOp.offsetTargetField() << "\n";

    // llvm::dbgs() << "processBlock(): relocationType() = " << relocOp.relocationType().str() << "\n";
    llvm::dbgs() << "processBlock(): relocationType() = " << static_cast<uint32_t>(relocOp.relocationType()) << "\n";

    llvm::dbgs() << "processBlock(): sourceSymbol() = " << relocOp.sourceSymbol() << "\n";
    llvm::dbgs() << "processBlock(): addend() = " << relocOp.addend() << "\n";

    int idx;
    for (idx = 0; idx < ELFSymbolIndex; idx++) {
        if (ELFSymbolValue[idx] == relocOp.sourceSymbol())
            break;
    }
    llvm::dbgs() << " Found idx = " << idx << "\n";

    auto relocationEntry = ((elf::writer::RelocationSection*)ELFSection[ELFSectionIndex])->addRelocationEntry();
    relocationEntry->setOffset(relocOp.offsetTargetField());
    relocationEntry->setSymbol(ELFSymbol[idx]);  // ELFSymbol[ELFSymbolIndex]);
    relocationEntry->setAddend(relocOp.addend());
    */
}

void vpux::ELF::CreateSymbolTableSectionOp::serialize(elf::Writer& elfWriter) {
    (void)elfWriter;

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
