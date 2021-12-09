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

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/frontend/VPUIP.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Translation.h>
// #include "llvm/Support/TypeSize.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include <llvm/Support/SourceMgr.h>

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

#include <fstream>
#include <iostream>

#include <cstdlib>

#include "llvm/Support/Debug.h"  // Alex

#include <elf/writer.hpp>  // 2021_10_19

using namespace vpux;

namespace {

//
// import-IE
//

mlir::OwningModuleRef importIE(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);

    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for IE IR, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto netFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (netFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for IE IR, not a file");
        return nullptr;
    }

    InferenceEngine::Core ieCore;
    InferenceEngine::CNNNetwork cnnNet;

    try {
        cnnNet = ieCore.ReadNetwork(netFileName.str());
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to open IE IR {0} : {1}", netFileName, ex.what());
        return nullptr;
    }

    mlir::OwningModuleRef module;

    try {
        mlir::DefaultTimingManager tm;
        auto rootTiming = tm.getRootScope();
        std::vector<vpux::PreProcessInfo> preProcInfo;
        module = IE::importNetwork(ctx, cnnNet, preProcInfo, false, rootTiming, false);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate IE IR {0} to MLIR : {1}", netFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-VPUIP
//

mlir::OwningModuleRef importVPUIP(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);

    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for blob, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto blobFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (blobFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for blob, not a file");
        return nullptr;
    }

    mlir::OwningModuleRef module;
    std::ifstream blobStream(blobFileName.str(), std::ios::binary);
    auto blob = std::vector<char>(std::istreambuf_iterator<char>(blobStream), std::istreambuf_iterator<char>());

    try {
        module = VPUIP::importBlob(ctx, blob);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate blob {0} to MLIR : {1}", blobFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// export-VPUIP
//

mlir::LogicalResult exportVPUIP(mlir::ModuleOp module, llvm::raw_ostream& output, StringRef /*outputFileName*/) {
    mlir::DefaultTimingManager tm;
    auto rootTiming = tm.getRootScope();
    std::vector<vpux::PreProcessInfo> preProcInfo;
    const auto buf = VPUIP::exportToBlobELF(module, rootTiming, preProcInfo);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

#define NUM_SECTIONS_MAX 100
#define NUM_SYMBOLS_MAX 10000
#define STR_ELF_SYMBOL "ELF.Symbol "

// Code taken from src/vpux_elf/example/simplewriter/simplewriter.cpp
static elf::Writer elfWriter;
static vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
static vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

void processRegion(mlir::Region& region);

// 2021_08_20: Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void printOperation(mlir::Operation* op) {
    // Print the operation itself and some of its properties
    llvm::dbgs() << "visiting op: '" << op->getName() << "' with " << op->getNumOperands() << " operands and "
                 << op->getNumResults() << " results\n";
    llvm::dbgs().flush();
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
        llvm::dbgs() << op->getAttrs().size() << " attributes:\n";
        for (mlir::NamedAttribute attr : op->getAttrs())
            llvm::dbgs() << " - '" << attr.first << "' : '" << attr.second << "'\n";
        llvm::dbgs().flush();
    }
}

// Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void processBlock(mlir::Block& block) {
    static struct ELFSectionAttributes {
        std::string sectionName;
        int sectionType;
        int sectionFlags;
        int sectionInfo;
        int sectionAddrAlignInfo;
        std::vector<char> serializedData;
    } sectionAttributes[NUM_SECTIONS_MAX];

    // elf::writer::BinaryDataSection<char>* ELFSection[NUM_SECTIONS_MAX];
    // static elf::writer::Section* ELFSection[NUM_SECTIONS_MAX];
    static std::vector<elf::writer::Section*> ELFSection;
    static int ELFSectionIndex = 0;
    // elf::writer::SymbolSection ELFSection2[NUM_SECTIONS_MAX];

    // static elf::writer::Symbol* ELFSymbol[NUM_SYMBOLS_MAX];
    static std::vector<std::pair<elf::writer::Symbol*, mlir::Value>> ELFSymbol;
    static int ELFSymbolIndex = 0;
    //
    // static mlir::Value ELFSymbolValue[NUM_SYMBOLS_MAX];

    // Print the block intrinsics properties (basically: argument list)
    llvm::dbgs() << "Entered processBlock(). Block with " << block.getNumArguments() << " arguments, "
                 << block.getNumSuccessors()
                 << " successors, and "
                 // Note, this `.size()` is traversing a linked-list and is O(n).
                 << block.getOperations().size() << " operations\n";
    llvm::dbgs().flush();

    // A block main role is to hold a list of Operations: let's recurse into
    //   printing each operation.
    for (mlir::Operation& op : block.getOperations()) {
        printOperation(&op);

        std::vector<char> buffer;

        // WORKING WELL but NOT Type-safe

        // See https://mlir.llvm.org/doxygen/classmlir_1_1OperationName.html
        //   and https://llvm.org/doxygen/classllvm_1_1StringRef.html
        std::string opName = op.getName().getStringRef().str();
        llvm::dbgs() << "  opName = " << opName << "\n";

        if (vpux::ELF::PutAnyOpInSectionOp putAnyOpInSectionOp = llvm::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op)) {
            llvm::dbgs() << "Found an ELF.PutAnyOpInSection operation\n";
            llvm::dbgs().flush();

            // mlir::Value putAnyOpValue = llvm::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op).inputArg(); // barrier();
            mlir::Value putAnyOpValue = putAnyOpInSectionOp.inputArg();  // barrier();

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
            } else if (vpux::VPUIPRegMapped::DeclareBufferOp putAnyOpOpDB =
                               llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareBufferOp>(putAnyOpOp)) {
                llvm::dbgs() << "    putAnyOpOpDB is of type vpux::VPUIPRegMapped::DeclareBufferOp\n";
                llvm::dbgs() << "    putAnyOpOpDB = " << putAnyOpOpDB << "\n";
                llvm::dbgs().flush();
            } else if (vpux::Const::DeclareOp putAnyOpOpDcl = llvm::dyn_cast<vpux::Const::DeclareOp>(putAnyOpOp)) {
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
            } else if (vpux::VPUIPRegMapped::NNDMAOp nndmaOp =
                               llvm::dyn_cast<vpux::VPUIPRegMapped::NNDMAOp>(putAnyOpOp)) {
                // llvm::dbgs() << "    putAnyOpOp is NOT of type vpux::ELF::SymbolOp\n";
                // llvm::dbgs().flush();

                llvm::dbgs() << "    putAnyOpOp is of type vpux::VPUIPRegMapped::NNDMAOp\n";
                llvm::dbgs().flush();
                // llvm::dbgs() << "Found a VPUIPRegMapped.NNDMA operation\n";
                // llvm::dbgs().flush();

                // llvm::cast<vpux::VPUIPRegMapped::NNDMAOp>(op).serialize(buffer);
                nndmaOp.serialize(buffer);

                llvm::dbgs() << "  Writing buffer for NNDMAOp to sectionAttributes[ELFSectionIndex].serializedData, "
                                "with ELFSectionIndex = "
                             << ELFSectionIndex << "\n";
                llvm::dbgs() << "    (buffer.size() = " << buffer.size() << ")\n";
                for (std::size_t i = 0; i < buffer.size(); i++) {
                    // 2021_10_19: DMATasksELFBLOB.push_back(buffer[i]);
                    // sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                    sectionAttributes[ELFSectionIndex].serializedData.push_back(i);
                }
            } else if (vpux::ELF::SymbolOp putAnyOpOpES = llvm::dyn_cast<vpux::ELF::SymbolOp>(putAnyOpOp)) {
                // IMPORTANT: We can have a memref or a Section
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
                // ELFSymbol[ELFSymbolIndex] =
                //        ((elf::writer::SymbolSection*)ELFSection[ELFSectionIndex])->addSymbolEntry();
                ELFSymbol.push_back(
                        std::make_pair(((elf::writer::SymbolSection*)ELFSection[ELFSectionIndex])->addSymbolEntry(),
                                       putAnyOpOpSymbol));  // 2021_12_08
                llvm::dbgs() << "processBlock(): ELFSymbol[ELFSymbolIndex].second = "
                             << ELFSymbol[ELFSymbolIndex].second << "\n";

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
                        // symbolNameStr = printStr.substr(posFound + strlen(STR_ELF_SYMBOL),
                        // printStr.size()); Note: For inputArg() instead of e.g. %arg0 it appears something
                        // like "<block argument> of type ..."
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
                ELFSymbol[ELFSymbolIndex].first->setName(symbolNameStr);
                // ELFSymbolValue[ELFSymbolIndex] = putAnyOpOpSymbol; // 2021_12_08

                ELFSymbol[ELFSymbolIndex].first->setValue(0);

                mlir::Type inputArgType = putAnyOpOpInputArg.getType();  // diOpInVec[idx].userType();
                int64_t inputArgTypeTotalSize = -1;
                if (mlir::ShapedType sType = inputArgType.dyn_cast<mlir::ShapedType>()) {
                    const Byte aByte = vpux::getTotalSize(sType);
                    // uint32_t tmp = static_cast<uint32_t>(tmpBit);
                    inputArgTypeTotalSize = aByte.count();
                    llvm::dbgs() << "  inputArgTypeTotalSize = " << inputArgTypeTotalSize << "\n";
                }
                ELFSymbol[ELFSymbolIndex].first->setSize(inputArgTypeTotalSize);

                llvm::dbgs() << "    ELFSymbolIndex = " << ELFSymbolIndex << "\n";
                ELFSymbolIndex++;
            } else {
                llvm::dbgs() << "    putAnyOpOp is none of the above (PutAnyOpInSectionOp)\n";
                llvm::dbgs() << "      op = " << op << "\n";
                llvm::dbgs().flush();
            }
        } else if (vpux::ELF::RelocOp relocOp = llvm::dyn_cast<vpux::ELF::RelocOp>(op)) {
            //             relocOp.serialize(/*elfWriter, */ ELFSection, ELFSectionIndex, ELFSymbol, ELFSymbolIndex);

            llvm::dbgs() << "processBlock(): Found ELF.RelocOp\n";
            llvm::dbgs().flush();

            // vpux::ELF::RelocOp opReloc = llvm::cast<vpux::ELF::RelocOp>(op);

            llvm::dbgs() << "processBlock(): offsetTargetField() = " << relocOp.offsetTargetField() << "\n";

            // llvm::dbgs() << "processBlock(): relocationType() = " << relocOp.relocationType().str() << "\n";
            llvm::dbgs() << "processBlock(): relocationType() = " << static_cast<uint32_t>(relocOp.relocationType())
                         << "\n";

            llvm::dbgs() << "processBlock(): sourceSymbol() = " << relocOp.sourceSymbol() << "\n";
            llvm::dbgs() << "processBlock(): addend() = " << relocOp.addend() << "\n";

            int idx;
            for (idx = 0; idx < ELFSymbolIndex; idx++) {
                // if (ELFSymbolValue[idx] == relocOp.sourceSymbol())
                if (ELFSymbol[idx].second == relocOp.sourceSymbol())
                    break;
            }
            llvm::dbgs() << "processBlock(): ELFSymbolIndex = " << ELFSymbolIndex << "\n";
            llvm::dbgs() << "processBlock(): Found idx = " << idx << "\n";
            llvm::dbgs() << "processBlock(): ELFSymbol[idx].second = " << ELFSymbol[idx].second << "\n";

            auto relocationEntry = ((elf::writer::RelocationSection*)ELFSection[ELFSectionIndex])->addRelocationEntry();
            relocationEntry->setOffset(relocOp.offsetTargetField());
            relocationEntry->setSymbol(ELFSymbol[idx].first);  // ELFSymbol[ELFSymbolIndex]);
            relocationEntry->setAddend(relocOp.addend());
        } else if (vpux::ELF::CreateSectionOp sectionOp = llvm::dyn_cast<vpux::ELF::CreateSectionOp>(op)) {
            llvm::dbgs() << "processBlock(): Found ELF.CreateSectionOp\n";
            llvm::dbgs().flush();

            // vpux::ELF::CreateSectionOp sectionOp = llvm::cast<vpux::ELF::CreateSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = sectionOp.secName().str();
            llvm::dbgs() << "processBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(sectionOp.secType());
            llvm::dbgs() << "processBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(sectionOp.secFlags());
            llvm::dbgs() << "processBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            sectionAttributes[ELFSectionIndex].sectionInfo = sectionOp.secInfo();
            llvm::dbgs() << "processBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = sectionOp.secAddrAlign();
            llvm::dbgs() << "processBlock(): secAddrAlign() = "
                         << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo << "\n";

            llvm::dbgs() << "  processBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            // See ...ELF/generated/ops.hpp.inc
            // MEGA TODO: check that the region contains only the same kind of Op (only NNDMAOp or only
            // ConfigureBarrierOp)
            mlir::Region& aRegion = sectionOp.aRegion();
            llvm::dbgs() << "processBlock(): Calling processRegion(aRegion)\n";
            processRegion(aRegion);

            llvm::dbgs() << "Creating section with name " << sectionAttributes[ELFSectionIndex].sectionName
                         << " and serializedData.size() = " << sectionAttributes[ELFSectionIndex].serializedData.size()
                         << ".\n";

            // ELFSection[ELFSectionIndex] = elfWriter.addBinaryDataSection<char>();
            ELFSection.push_back(elfWriter.addBinaryDataSection<char>(
                    sectionAttributes[ELFSectionIndex].sectionName));  // 2021_12_08 // 2021_12_09
            // ELFSection[ELFSectionIndex]->setName(sectionAttributes[ELFSectionIndex].sectionName);
            // ELFSection[idx]->set_type(sectionAttributes[idx].sectionType);
            // ELFSection[idx]->setType(sectionAttributes[idx].sectionType); // TODO
            // ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
            ELFSection[ELFSectionIndex]->setFlags(sectionAttributes[ELFSectionIndex].sectionFlags);
            // ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
            ELFSection[ELFSectionIndex]->setAddrAlign(sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo);

            // ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
            //                          sectionAttributes[idx].serializedData.size());
            // for (std::size_t i = 0; i < sectionAttributes[idx].serializedData.size(); i++)
            //    ELFSection[idx]->appendData(sectionAttributes[idx].serializedData[i]);
            ((elf::writer::BinaryDataSection<char>*)ELFSection[ELFSectionIndex])
                    ->appendData(sectionAttributes[ELFSectionIndex].serializedData.data(),
                                 sectionAttributes[ELFSectionIndex].serializedData.size());

            llvm::dbgs() << "  processBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELF::Section, etc)
        } else if (vpux::ELF::CreateSymbolTableSectionOp sectionOp =
                           llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(op)) {
            llvm::dbgs() << "processBlock(): Found ELF.CreateSymbolTableSection\n";
            llvm::dbgs().flush();

            // vpux::ELF::CreateSymbolTableSectionOp sectionOp = llvm::cast<vpux::ELF::CreateSymbolTableSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = sectionOp.secName().str();
            // sectionAttributes[ELFSectionIndex].sectionName = ".symTab_ELF_MLIR";
            llvm::dbgs() << "processBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            // sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(sectionOp.secType());
            sectionAttributes[ELFSectionIndex].sectionType = 2;  // SYMTAB
            llvm::dbgs() << "processBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            // sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(sectionOp.secFlags());
            sectionAttributes[ELFSectionIndex].sectionFlags = 4;  // TODO
            llvm::dbgs() << "processBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            // sectionAttributes[ELFSectionIndex].sectionInfo = sectionOp.secInfo();
            sectionAttributes[ELFSectionIndex].sectionInfo = 1;  // TODO
            llvm::dbgs() << "processBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            // sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = sectionOp.secAddrAlign();
            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = 64;  // TODO
            llvm::dbgs() << "processBlock(): secAddrAlign() = "
                         << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo << "\n";

            llvm::dbgs() << "  processBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            llvm::dbgs() << "Creating symbol table section with name " << sectionAttributes[ELFSectionIndex].sectionName
                         << " and serializedData.size() = " << sectionAttributes[ELFSectionIndex].serializedData.size()
                         << ".\n";

            // ELFSection[ELFSectionIndex] = elfWriter.addBinaryDataSection<char>();
            // ELFSection[ELFSectionIndex] = elfWriter.addSymbolSection();
            ELFSection.push_back(elfWriter.addSymbolSection(
                    sectionAttributes[ELFSectionIndex].sectionName));  // 2021_12_08 // 2021_12_09
            // ELFSection[ELFSectionIndex]->setName(sectionAttributes[ELFSectionIndex].sectionName);
            // <<error: ‘class elf::writer::SymbolSection’ has no member named ‘setType’:>>
            // ((elf::writer::SymbolSection*)ELFSection[ELFSectionIndex])
            //        ->setType(sectionAttributes[ELFSectionIndex].sectionType); // TODO
            // ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
            ELFSection[ELFSectionIndex]->setFlags(sectionAttributes[ELFSectionIndex].sectionFlags);
            // ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
            ELFSection[ELFSectionIndex]->setAddrAlign(sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo);

            // ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
            //                          sectionAttributes[idx].serializedData.size());
            // for (std::size_t i = 0; i < sectionAttributes[idx].serializedData.size(); i++)
            //    ELFSection[idx]->appendData(sectionAttributes[idx].serializedData[i]);
            /*
            ((elf::writer::BinaryDataSection<char>*)ELFSection[ELFSectionIndex])
                    ->appendData(sectionAttributes[ELFSectionIndex].serializedData.data(),
                              sectionAttributes[ELFSectionIndex].serializedData.size());
            */

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = sectionOp.aRegion();
            llvm::dbgs() << "processBlock(): Calling processRegion(aRegion)\n";
            processRegion(aRegion);

            llvm::dbgs() << "  processBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELF::Section, etc)
        } else if (vpux::ELF::CreateRelocationSectionOp sectionOp =
                           llvm::dyn_cast<vpux::ELF::CreateRelocationSectionOp>(op)) {
            llvm::dbgs() << "processBlock(): Found ELF.CreateRelocationSection\n";
            llvm::dbgs().flush();

            // vpux::ELF::CreateRelocationSectionOp sectionOp = llvm::cast<vpux::ELF::CreateRelocationSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = sectionOp.secName().str();
            // sectionAttributes[ELFSectionIndex].sectionName = ".rela_ELF_MLIR";
            llvm::dbgs() << "processBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            llvm::dbgs() << "processBlock(): sourceSymbolTableSection() = " << sectionOp.sourceSymbolTableSection()
                         << "\n";
            llvm::dbgs() << "processBlock(): targetSection() = " << sectionOp.targetSection() << "\n";

            // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
            mlir::Operation* sstsOp = sectionOp.sourceSymbolTableSection().getDefiningOp();
            llvm::dbgs() << "    *sstsOp = " << *sstsOp << "\n";
            llvm::dbgs().flush();
            //
            vpux::ELF::CreateSymbolTableSectionOp sstsOp2 =
                    llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(sstsOp);
            llvm::dbgs() << "processBlock(): sourceSymbolTableSection().name = " << sstsOp2.secName().str() << "\n";

            std::string sourceSymbolTableSectionName = sstsOp2.secName().str();

            mlir::Operation* tsOp = sectionOp.targetSection().getDefiningOp();
            llvm::dbgs() << "    *sstsOp = " << *tsOp << "\n";
            llvm::dbgs().flush();
            //
            vpux::ELF::CreateSectionOp tsOp2 = llvm::dyn_cast<vpux::ELF::CreateSectionOp>(tsOp);
            llvm::dbgs() << "processBlock(): targetSection().name = " << tsOp2.secName().str() << "\n";

            std::string targetSectionName = tsOp2.secName().str();

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            // sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(sectionOp.secType());
            sectionAttributes[ELFSectionIndex].sectionType = 2;
            llvm::dbgs() << "processBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(sectionOp.secFlags());
            // sectionAttributes[ELFSectionIndex].sectionFlags = 4;  // TODO
            llvm::dbgs() << "processBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            // sectionAttributes[ELFSectionIndex].sectionInfo = sectionOp.secInfo();
            sectionAttributes[ELFSectionIndex].sectionInfo = 1;  // TODO
            llvm::dbgs() << "processBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            // sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = sectionOp.secAddrAlign();
            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = 64;  // TODO
            llvm::dbgs() << "processBlock(): secAddrAlign() = "
                         << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo << "\n";

            llvm::dbgs() << "  processBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            // Search in sectionAttributes for the names of the sections.
            int idx1;
            int idx2;
            for (idx1 = 0; idx1 < ELFSectionIndex; idx1++) {
                if (sectionAttributes[idx1].sectionName == sourceSymbolTableSectionName)
                    break;
            }
            llvm::dbgs() << "idx1 = " << idx1 << "\n";

            for (idx2 = 0; idx2 < ELFSectionIndex; idx2++) {
                if (sectionAttributes[idx2].sectionName == targetSectionName)
                    break;
            }
            llvm::dbgs() << "idx2 = " << idx2 << "\n";

            llvm::dbgs() << "Creating relocation section with name " << sectionAttributes[ELFSectionIndex].sectionName
                         << " and serializedData.size() = " << sectionAttributes[ELFSectionIndex].serializedData.size()
                         << ".\n";

            // ELFSection[ELFSectionIndex] = elfWriter.addRelocationSection(); // 2021_12_08
            ELFSection.push_back(elfWriter.addRelocationSection(
                    sectionAttributes[ELFSectionIndex].sectionName));  // 2021_12_08 // 2021_12_09
            // auto relocation = elfWriter.addRelocationSection();
            // ELFSection[ELFSectionIndex] = elfWriter.addBinaryDataSection<char>();
            // ELFSection[ELFSectionIndex]->setName(sectionAttributes[ELFSectionIndex].sectionName); // 2021_12_09
            // ELFSection[idx]->set_type(sectionAttributes[idx].sectionType);
            // ELFSection[idx]->setType(sectionAttributes[idx].sectionType); // TODO
            // ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
            ELFSection[ELFSectionIndex]->setFlags(sectionAttributes[ELFSectionIndex].sectionFlags);
            // ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
            ELFSection[ELFSectionIndex]->setAddrAlign(sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo);

            // ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
            //                          sectionAttributes[idx].serializedData.size());
            // for (std::size_t i = 0; i < sectionAttributes[idx].serializedData.size(); i++)
            //    ELFSection[idx]->appendData(sectionAttributes[idx].serializedData[i]);
            /*
            ((elf::writer::BinaryDataSection<char>*)ELFSection[ELFSectionIndex])
                    ->appendData(sectionAttributes[ELFSectionIndex].serializedData.data(),
                              sectionAttributes[ELFSectionIndex].serializedData.size());
            */

            ((elf::writer::RelocationSection*)ELFSection[ELFSectionIndex])
                    ->setSymbolTable((const elf::writer::SymbolSection*)ELFSection[idx1]);
            ((elf::writer::RelocationSection*)ELFSection[ELFSectionIndex])->setSectionToPatch(ELFSection[idx2]);

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = sectionOp.aRegion();
            llvm::dbgs() << "processBlock(): Calling processRegion(aRegion)\n";
            processRegion(aRegion);

            llvm::dbgs() << "  processBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELF::Section, etc)
        } else {
            llvm::dbgs() << "    op is none of the above\n";
            llvm::dbgs() << "      op = " << op << "\n";
            llvm::dbgs().flush();
        }

        llvm::dbgs().flush();
    }
}

// 2021_08_20: Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void processRegion(mlir::Region& region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::dbgs() << "Entered processRegion(). Region with " << region.getBlocks().size() << " blocks\n";
    // Not working << region << "\n";

    for (mlir::Block& block : region.getBlocks())
        processBlock(block);
}

mlir::LogicalResult exportVPUIPRegMappedAndELF(mlir::ModuleOp module, llvm::raw_ostream& output,
                                               StringRef outputFileName) {
    (void)output;

    llvm::dbgs() << "Alex: Entered exportVPUIPRegMappedAndELF()\n";
    llvm::dbgs() << "exportVPUIPELF(): module->getName() = " << module->getName() << "\n";
    llvm::dbgs() << "exportVPUIPELF(): outputFileName = " << outputFileName << "\n";

    // Code taken from src/vpux_elf/example/simplewriter/simplewriter.cpp
    // elf::Writer elf;

    // writer.create(ELFCLASS64, ELFDATA2LSB); // TODO

    // writer.set_os_abi(ELFOSABI_LINUX); // TODO
    // writer.set_type(ET_EXEC); // TODO
    // writer.set_machine( EM_X86_64 );
    // writer.set_machine(EM_res035); // TODO

    for (mlir::Operation& op : module) {
        if (vpux::IE::CNNNetworkOp cnnOp = llvm::dyn_cast<vpux::IE::CNNNetworkOp>(op)) {
            llvm::dbgs() << "Found a IE::CNNNetworkOp operation\n";
            llvm::dbgs().flush();

            diOpInVec = cnnOp.getInputsInfo();
            llvm::dbgs() << "  diOpInVec.size() = " << diOpInVec.size() << "\n";
            llvm::dbgs().flush();
            unsigned int idx;
            for (idx = 0; idx < diOpInVec.size(); idx++) {
                llvm::dbgs() << "  diOpInVec[" << idx << "] = " << diOpInVec[idx] << "\n";
                llvm::dbgs() << "  diOpInVec[" << idx << "].name() = " << diOpInVec[idx].name() << "\n";
                llvm::dbgs() << "  diOpInVec[" << idx << "].userType() = " << diOpInVec[idx].userType() << "\n";
                llvm::dbgs().flush();
            }

            diOpOutVec = cnnOp.getOutputsInfo();
            llvm::dbgs() << "  diOpOutVec.size() = " << diOpOutVec.size() << "\n";
            llvm::dbgs().flush();
            for (idx = 0; idx < diOpOutVec.size(); idx++) {
                llvm::dbgs() << "  diOpOutVec[" << idx << "] = " << diOpOutVec[idx] << "\n";
                llvm::dbgs() << "  diOpOutVec[" << idx << "].name() = " << diOpOutVec[idx].name() << "\n";
                llvm::dbgs() << "  diOpOutVec[" << idx << "].userType() = " << diOpOutVec[idx].userType() << "\n";
                llvm::dbgs().flush();
            }
        } else if (mlir::isa<mlir::FuncOp>(op)) {
            mlir::FuncOp funcOp = llvm::cast<mlir::FuncOp>(op);  // use maybe mlir::cast
            // <<error: no match for ‘operator<<’>>: llvm::dbgs() << "funcOp.getArguments() = " << funcOp.getArguments()
            // << "\n";

            /*
            // <<error: no match for ‘operator<<’>>: llvm::dbgs() << "funcOp.getArgAttrs(0) = " << funcOp.getArgAttrs(0)
            // << "\n";
            unsigned int idx;
            //
            ArrayRef<mlir::NamedAttribute> argAttrs0 = funcOp.getArgAttrs(0);
            for (idx = 0; idx < argAttrs0.size(); idx++) {
                llvm::dbgs() << "argAttrs0.first = " << argAttrs0[idx].first << "\n";
                llvm::dbgs() << "argAttrs0.second = " << argAttrs0[idx].second << "\n";
            }
            //
            ArrayRef<mlir::NamedAttribute> argAttrs1 = funcOp.getArgAttrs(1);
            for (idx = 0; idx < argAttrs1.size(); idx++) {
                llvm::dbgs() << "argAttrs1.first = " << argAttrs1[idx].first << "\n";
                llvm::dbgs() << "argAttrs1.second = " << argAttrs1[idx].second << "\n";
            }
            */

            // See https://mlir.llvm.org/doxygen/FunctionSupport_8h_source.html
            unsigned int numArgs = funcOp.getNumArguments();
            llvm::dbgs() << "numArgs = " << numArgs << "\n";
            //
            for (unsigned int idx = 0; idx < numArgs; idx++) {
                llvm::dbgs() << "funcOp.getArgument(" << idx << ") = " << funcOp.getArgument(idx) << "\n";
            }

            llvm::dbgs()
                    // printIndent()
                    << "  Function name: " << funcOp.getName() << "\n";

            // I am not using FuncOp::walk() because we have subregions
            processRegion(*(funcOp.getCallableRegion()));
        }
    }

    // elfWriter.write("vpux_elf_MTL");
    const auto elfBlob = elfWriter.generateELF();  // 2021_12_09

    std::ofstream stream("vpux_elf_MTL", std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(elfBlob.data()), elfBlob.size());

    // llvm::dbgs() << "When exiting exportVPUIPELF(): ELFSectionIndex = " << ELFSectionIndex << "\n";
    // llvm::dbgs().flush();

    return mlir::success();
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        mlir::TranslateToMLIRRegistration("import-IE", importIE);
        mlir::TranslateToMLIRRegistration("import-HWTEST", importHWTEST);
        mlir::TranslateToMLIRRegistration("import-VPUIP", importVPUIP);
        mlir::TranslateFromMLIRRegistration("export-VPUIP", exportVPUIP, registerDialects);
        mlir::TranslateFromMLIRRegistration("export-VPUIP-ELF", exportVPUIPRegMappedAndELF, registerDialects);

        return mlir::asMainReturnCode(mlir::mlirTranslateMain(argc, argv, "VPUX Translation Testing Tool"));
    } catch (const std::exception& e) {
        llvm::dbgs() << "Alex: main(): caught exception\n";

        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
