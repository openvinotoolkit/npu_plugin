//
// Copyright 2021 Intel Corporation.
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
#include "vpux/hwtest/hwtest.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Translation.h>

#include <llvm/Support/SourceMgr.h>

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

#include <fstream>
#include <iostream>

#include <cstdlib>

//#include <elfio/elfio.hpp>       // Alex
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
        module = IE::importNetwork(ctx, cnnNet, false, rootTiming);
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

mlir::LogicalResult exportVPUIP(mlir::ModuleOp module, llvm::raw_ostream& output) {
    mlir::DefaultTimingManager tm;
    auto rootTiming = tm.getRootScope();
    const auto buf = VPUIP::exportToBlob(module, rootTiming);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

/*
  /// TODO: Alex

  // Code from thirdparty/llvm-project/mlir/test/lib/IR/TestPrintNesting.cpp
  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }
*/

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

int ELFSectionIndex = 0;  // TODO: change accordingly

// std::vector<char> DMATasksELFBLOB;
// std::vector<char> ConfigureBarriersELFBLOB;

#define NUM_SECTIONS_MAX 100
struct ELFSectionAttributes {
    std::string sectionName;
    int sectionType;
    int sectionFlags;
    int sectionInfo;
    int sectionAddrAlignInfo;
    std::vector<char> serializedData;
} sectionAttributes[NUM_SECTIONS_MAX];
#ifdef USE_ELFIO
ELFIO::section* ELFSection[NUM_SECTIONS_MAX];
#endif
elf::writer::BinaryDataSection<char>* ELFSection[NUM_SECTIONS_MAX];
// Code taken from src/vpux_elf/example/simplewriter/simplewriter.cpp
elf::Writer elf;

void printRegion(mlir::Region& region);

// Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void printBlock(mlir::Block& block) {
    // Print the block intrinsics properties (basically: argument list)
    llvm::dbgs() << "Entered printBlock(). Block with " << block.getNumArguments() << " arguments, "
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

        if (opName.find("NNDMA") != std::string::npos) {
            llvm::dbgs() << "Found a VPUIPRegMapped.NNDMA operation\n";

            llvm::cast<vpux::VPUIPRegMapped::NNDMAOp>(op).serialize(buffer);

            llvm::dbgs() << "  Writing buffer for NNDMAOp to sectionAttributes[ELFSectionIndex].serializedData, "
                            "with ELFSectionIndex = "
                         << ELFSectionIndex << "\n";
            llvm::dbgs() << "    (buffer.size() = " << buffer.size() << ")\n";
            for (std::size_t i = 0; i < buffer.size(); i++) {
                // 2021_10_19: DMATasksELFBLOB.push_back(buffer[i]);
                // sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                sectionAttributes[ELFSectionIndex].serializedData.push_back(i);
            }
        }
        /*
        else
        if (opName == "VPUIPRegMapped.ConfigureBarrier") {
            // Just printing the Operation
            //auto aCfgBarrier = llvm::cast<vpux::VPUIPRegMapped::ConfigureBarrierOp>(op);
            vpux::VPUIPRegMapped::ConfigureBarrierOp aCfgBarrier =
        llvm::cast<vpux::VPUIPRegMapped::ConfigureBarrierOp>(op);

            llvm::dbgs() << "    aCfgBarrier = " << aCfgBarrier << "\n";
            //llvm::dbgs().flush();
        }
        */
        else if (opName.find("PutAnyOpInSection") != std::string::npos) {
            mlir::Value aPutAnyOpValue = llvm::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op).inputArg();  // barrier();
            llvm::dbgs() << "    aPutAnyOpValue = " << aPutAnyOpValue << "\n";
            // See https://llvm.org/doxygen/classllvm_1_1raw__ostream.html
            llvm::dbgs().flush();

            // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
            mlir::Operation* aPutAnyOpOp = aPutAnyOpValue.getDefiningOp();
            llvm::dbgs() << "    *aPutAnyOpOp = " << *aPutAnyOpOp << "\n";
            llvm::dbgs().flush();

            vpux::VPUIPRegMapped::ConfigureBarrierOp aPutAnyOpOp2 =
                    llvm::dyn_cast<vpux::VPUIPRegMapped::ConfigureBarrierOp>(aPutAnyOpOp);
            if (!aPutAnyOpOp2) {
                llvm::dbgs() << "    aPutAnyOpOp2 is NOT of type vpux::VPUIPRegMapped::ConfigureBarrierOp\n";
                llvm::dbgs().flush();

                vpux::VPUIPRegMapped::DeclareTensorOp aPutAnyOpOp2dt =
                        llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareTensorOp>(aPutAnyOpOp);
                if (!aPutAnyOpOp2dt) {
                    llvm::dbgs() << "    aPutAnyOpOp2dt is NOT of type vpux::VPUIPRegMapped::DeclareTensorOp\n";
                    llvm::dbgs().flush();

#if 0
                    vpux::ELF::SymbolOp aPutAnyOpOp2dt3 = llvm::dyn_cast<vpux::ELF::SymbolOp>(aPutAnyOpOp);
                    // IMPORTANT: We can have a memref or a Section
                    if (!aPutAnyOpOp2dt3) {
                        llvm::dbgs() << "    aPutAnyOpOp2dt3 is NOT of type vpux::ELF::SymbolOp\n";
                        llvm::dbgs().flush();
                    } else {
                        llvm::dbgs() << "    aPutAnyOpOp2dt3 is of type vpux::ELF::SymbolOp\n";
                        llvm::dbgs().flush();

                        llvm::dbgs() << "    aPutAnyOpOp2dt3 = " << aPutAnyOpOp2dt3 << "\n";

                        mlir::Value aPutAnyOpValue4 = aPutAnyOpOp2dt3.inputArg();

                        // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
                        mlir::Operation* aPutAnyOpOp2dt3_op = aPutAnyOpValue4.getDefiningOp();
                        llvm::dbgs() << "    *aPutAnyOpOp2dt3_op = " << *aPutAnyOpOp2dt_op << "\n";
                        llvm::dbgs().flush();

                        /*
                        llvm::dbgs() << "    aPutAnyOpValue = " << aPutAnyOpValue << "\n";
                        // See https://llvm.org/doxygen/classllvm_1_1raw__ostream.html
                        llvm::dbgs().flush();

                        // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
                        mlir::Operation* aPutAnyOpOp = aPutAnyOpValue.getDefiningOp();
                        llvm::dbgs() << "    *aPutAnyOpOp = " << *aPutAnyOpOp << "\n";
                        llvm::dbgs().flush();

                        vpux::VPUIPRegMapped::ConfigureBarrierOp aPutAnyOpOp2 =
                                llvm::dyn_cast<vpux::VPUIPRegMapped::ConfigureBarrierOp>(aPutAnyOpOp);
                        if (!aPutAnyOpOp2) {
                            llvm::dbgs() << "    aPutAnyOpOp2 is NOT of type
                        vpux::VPUIPRegMapped::ConfigureBarrierOp\n"; llvm::dbgs().flush();
                        }
                        */
                    }
#endif

                    vpux::Const::DeclareOp aPutAnyOpOp2dt2 = llvm::dyn_cast<vpux::Const::DeclareOp>(aPutAnyOpOp);
                    if (!aPutAnyOpOp2dt2) {
                        llvm::dbgs() << "    aPutAnyOpOp2dt2 is NOT of type vpux::Const::DeclareOp\n";
                        llvm::dbgs().flush();
                    } else {
                        llvm::dbgs() << "    aPutAnyOpOp2dt2 is of type vpux::Const::DeclareOp\n";
                        llvm::dbgs() << "    aPutAnyOpOp2dt2 = " << aPutAnyOpOp2dt2 << "\n";
                        llvm::dbgs().flush();

                        // aPutAnyOpOp2dt2.content();
                        // Defined in include/vpux/compiler/dialect/const/utils/content.hpp
                        // getValues()
                        /*
                        for (std::size_t i = 0; i < buffer.size(); i++) {
                            // sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                            sectionAttributes[ELFSectionIndex].serializedData.push_back(i);  // 2021_10_19
                        }
                        */

                        aPutAnyOpOp2dt2.serialize(buffer);

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
                    }
                } else {
                    llvm::dbgs() << "    aPutAnyOpOp2dt is of type vpux::VPUIPRegMapped::DeclareTensorOp\n";
                    llvm::dbgs() << "    aPutAnyOpOp2dt = " << aPutAnyOpOp2dt << "\n";
                    llvm::dbgs().flush();
                }
            } else {
                llvm::dbgs() << "    aPutAnyOpOp2 is of type vpux::VPUIPRegMapped::ConfigureBarrierOp\n";
                llvm::dbgs() << "    aPutAnyOpOp2 = " << aPutAnyOpOp2 << "\n";
                llvm::dbgs().flush();
                aPutAnyOpOp2.serialize(buffer);

                llvm::dbgs() << "  Writing buffer for ConfigureBarrierOp to "
                                "sectionAttributes[ELFSectionIndex].serializedData, with ELFSectionIndex = "
                             << ELFSectionIndex << "\n";
                llvm::dbgs() << "    (buffer.size() = " << buffer.size() << ")\n";
                for (std::size_t i = 0; i < buffer.size(); i++) {
                    // 2021_10_19: ConfigureBarriersELFBLOB.push_back(buffer[i]);
                    // sectionAttributes[ELFSectionIndex].serializedData.push_back(buffer[i]);  // 2021_10_19
                    sectionAttributes[ELFSectionIndex].serializedData.push_back(i);  // 2021_10_19
                }
            }

        } else if (opName == "ELF.Reloc") {
            llvm::dbgs() << "printBlock(): Found ELF.RelocOp\n";

            vpux::ELF::RelocOp opReloc = llvm::cast<vpux::ELF::RelocOp>(op);

            llvm::dbgs() << "printBlock(): offsetTargetField() = " << opReloc.offsetTargetField() << "\n";

            // llvm::dbgs() << "printBlock(): relocationType() = " << opReloc.relocationType().str() << "\n";
            llvm::dbgs() << "printBlock(): relocationType() = " << static_cast<uint32_t>(opReloc.relocationType())
                         << "\n";

            llvm::dbgs() << "printBlock(): sourceSymbol() = " << opReloc.sourceSymbol() << "\n";
            llvm::dbgs() << "printBlock(): addend() = " << opReloc.addend() << "\n";
        } else if (opName.find("CreateSection") != std::string::npos) {
            llvm::dbgs() << "printBlock(): Found ELF.CreateSectionOp\n";

            vpux::ELF::CreateSectionOp opSection = llvm::cast<vpux::ELF::CreateSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = opSection.secName().str();
            llvm::dbgs() << "printBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(opSection.secType());
            llvm::dbgs() << "printBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(opSection.secFlags());
            llvm::dbgs() << "printBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            sectionAttributes[ELFSectionIndex].sectionInfo = opSection.secInfo();
            llvm::dbgs() << "printBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = opSection.secAddrAlign();
            llvm::dbgs() << "printBlock(): secAddrAlign() = " << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo
                         << "\n";

            llvm::dbgs() << "  printBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = opSection.aRegion();
            llvm::dbgs() << "printBlock(): Calling printRegion(aRegion)\n";
            printRegion(aRegion);

            llvm::dbgs() << "Creating section with name " << sectionAttributes[ELFSectionIndex].sectionName
                         << " and serializedData.size() = " << sectionAttributes[ELFSectionIndex].serializedData.size()
                         << ".\n";

            ELFSection[ELFSectionIndex] = elf.addBinaryDataSection<char>();
            ELFSection[ELFSectionIndex]->setName(sectionAttributes[ELFSectionIndex].sectionName);
            // ELFSection[idx]->set_type(sectionAttributes[idx].sectionType);
            // ELFSection[idx]->setType(sectionAttributes[idx].sectionType); // TODO
            // ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
            ELFSection[ELFSectionIndex]->setFlags(sectionAttributes[ELFSectionIndex].sectionFlags);
            // ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
            ELFSection[ELFSectionIndex]->setAddrAlign(sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo);

            // ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
            //                          sectionAttributes[idx].serializedData.size());
            // for (std::size_t i = 0; i < sectionAttributes[idx].serializedData.size(); i++)
            //    ELFSection[idx]->addData(sectionAttributes[idx].serializedData[i]);
            ELFSection[ELFSectionIndex]->addData(sectionAttributes[ELFSectionIndex].serializedData.data(),
                                                 sectionAttributes[ELFSectionIndex].serializedData.size());

            llvm::dbgs() << "  printBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELFIO::section, etc)
        } else if (opName.find("CreateSymbolTableSection") != std::string::npos) {
            llvm::dbgs() << "printBlock(): Found ELF.CreateSymbolTableSection\n";

            // vpux::ELF::CreateSectionOp opSection = llvm::cast<vpux::ELF::CreateSectionOp>(op);
            vpux::ELF::CreateSymbolTableSectionOp opSection = llvm::cast<vpux::ELF::CreateSymbolTableSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = opSection.secName().str();
            // sectionAttributes[ELFSectionIndex].sectionName = ".symTab_ELF_MLIR";
            llvm::dbgs() << "printBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            // sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(opSection.secType());
            sectionAttributes[ELFSectionIndex].sectionType = 2;
            llvm::dbgs() << "printBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            // sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(opSection.secFlags());
            sectionAttributes[ELFSectionIndex].sectionFlags = 4;  // TODO
            llvm::dbgs() << "printBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            // sectionAttributes[ELFSectionIndex].sectionInfo = opSection.secInfo();
            sectionAttributes[ELFSectionIndex].sectionInfo = 1;  // TODO
            llvm::dbgs() << "printBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            // sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = opSection.secAddrAlign();
            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = 64;  // TODO
            llvm::dbgs() << "printBlock(): secAddrAlign() = " << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo
                         << "\n";

            llvm::dbgs() << "  printBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = opSection.aRegion();
            llvm::dbgs() << "printBlock(): Calling printRegion(aRegion)\n";
            printRegion(aRegion);

            llvm::dbgs() << "  printBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELFIO::section, etc)
        } else if (opName.find("CreateRelocationSection") != std::string::npos) {
            llvm::dbgs() << "printBlock(): Found ELF.CreateRelocationSection\n";

            // vpux::ELF::CreateSectionOp opSection = llvm::cast<vpux::ELF::CreateSectionOp>(op);
            vpux::ELF::CreateRelocationSectionOp opSection = llvm::cast<vpux::ELF::CreateRelocationSectionOp>(op);

            sectionAttributes[ELFSectionIndex].sectionName = opSection.secName().str();
            // sectionAttributes[ELFSectionIndex].sectionName = ".rela_ELF_MLIR";
            llvm::dbgs() << "printBlock(): secName() = " << sectionAttributes[ELFSectionIndex].sectionName << "\n";

            llvm::dbgs() << "printBlock(): sourceSymbolTableSection() = " << opSection.sourceSymbolTableSection()
                         << "\n";
            llvm::dbgs() << "printBlock(): targetSection() = " << opSection.targetSection() << "\n";

            // See https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
            mlir::Operation* sstsOp = opSection.sourceSymbolTableSection().getDefiningOp();
            llvm::dbgs() << "    *sstsOp = " << *sstsOp << "\n";
            llvm::dbgs().flush();
            //
            vpux::ELF::CreateSymbolTableSectionOp sstsOp2 =
                    llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(sstsOp);
            llvm::dbgs() << "printBlock(): sourceSymbolTableSection().name = " << sstsOp2.secName().str() << "\n";
            // MEGA-TODO: search for index in ELFSection[].

            mlir::Operation* tsOp = opSection.targetSection().getDefiningOp();
            llvm::dbgs() << "    *sstsOp = " << *tsOp << "\n";
            llvm::dbgs().flush();
            //
            vpux::ELF::CreateSectionOp tsOp2 = llvm::dyn_cast<vpux::ELF::CreateSectionOp>(tsOp);
            llvm::dbgs() << "printBlock(): targetSection().name = " << tsOp2.secName().str() << "\n";

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            // sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(opSection.secType());
            sectionAttributes[ELFSectionIndex].sectionType = 2;
            llvm::dbgs() << "printBlock(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(opSection.secFlags());
            // sectionAttributes[ELFSectionIndex].sectionFlags = 4;  // TODO
            llvm::dbgs() << "printBlock(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags << "\n";
            // << std::hex // small-TODO: use write_hex()

            // sectionAttributes[ELFSectionIndex].sectionInfo = opSection.secInfo();
            sectionAttributes[ELFSectionIndex].sectionInfo = 1;  // TODO
            llvm::dbgs() << "printBlock(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            // sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = opSection.secAddrAlign();
            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = 64;  // TODO
            llvm::dbgs() << "printBlock(): secAddrAlign() = " << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo
                         << "\n";

            llvm::dbgs() << "  printBlock(): ELFSectionIndex = " << ELFSectionIndex << "\n";

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = opSection.aRegion();
            llvm::dbgs() << "printBlock(): Calling printRegion(aRegion)\n";
            printRegion(aRegion);

            //
            // Relocations
            //

            /*
            auto relocation = elf.addRelocationSection();
            // MEGA-TODO: relocationEntry->setSymbolTable(ELFSection[]);
            // MEGA-TODO: relocationEntry->setSectionToPatch(ELFSection[]);
            relocation->setName(sectionAttributes[ELFSectionIndex].sectionName);
            */

            llvm::dbgs() << "  printBlock(): Before increment ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELFIO::section, etc)
        }

        // llvm::dbgs() << "DMATasksELFBLOB.size() = " << DMATasksELFBLOB.size() << "\n";
        // llvm::dbgs() << "ConfigureBarriersELFBLOB.size() = " << ConfigureBarriersELFBLOB.size() << "\n";
        llvm::dbgs().flush();
    }
}

// 2021_08_20: Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void printRegion(mlir::Region& region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::dbgs()
            // printIndent()
            << "Entered printRegion(). Region with " << region.getBlocks().size() << " blocks:\n";

    // auto indent = pushIndent();
    for (mlir::Block& block : region.getBlocks())
        printBlock(block);
}

mlir::LogicalResult exportVPUIPELF(mlir::ModuleOp module, llvm::raw_ostream& output) {
    llvm::dbgs() << "Alex: Entered exportVPUIPELF()\n";
    llvm::dbgs() << "exportVPUIPELF(): module->getName() = " << module->getName() << "\n";

    // Code taken from src/vpux_elf/example/simplewriter/simplewriter.cpp
    // elf::Writer elf;

    // writer.create(ELFCLASS64, ELFDATA2LSB); // TODO

    // writer.set_os_abi(ELFOSABI_LINUX); // TODO
    // writer.set_type(ET_EXEC); // TODO
    // writer.set_machine( EM_X86_64 );
    // writer.set_machine(EM_res035); // TODO

    //
    // SymbolSection
    //

    auto symbolSection = elf.addSymbolSection();
    symbolSection->setName(".symtab");

    auto inputSym = symbolSection->addSymbolEntry();
    inputSym->setName(".input");

    auto outputSym = symbolSection->addSymbolEntry();
    outputSym->setName(".output");

    //
    // Weights
    //

#if 0
    auto weights = elf.addSegment();
    weights->setType(elf::PT_LOAD);
    weights->addData("11111", 5);
#endif
    // Note: this section has type PROGBITS.
    // See Table from slide 26 of Andrew Bakalin's presentation ELF PoC_new.pptx
    // elf::writer::BinaryDataSection<char>* weights_sec = elf.addBinaryDataSection<char>();
    // weights_sec->setName(".data.Weights");

    for (mlir::Operation& op : module) {
        if (mlir::isa<mlir::FuncOp>(op)) {
            auto func = llvm::cast<mlir::FuncOp>(op);  // use maybe mlir::cast
            llvm::dbgs()
                    // printIndent()
                    << "  Function name: " << func.getName() << "\n";

            printRegion(*(func.getCallableRegion()));
        }
    }

    /*
    struct DMATask {
        int x;
        double y;
    };

    // auto dmaTasks = elf.addBinaryDataSection<DMATask>();
    auto dmaTasks = elf.addBinaryDataSection<char>();
    dmaTasks->setName(".text.dmaTasks");
    dmaTasks->setAddrAlign(64);
    // dmaTasks->addData(DMATask());
    for (int i = 0; i < 80; i++)
        dmaTasks->addData(i);

    elf::writer::BinaryDataSection<char>* barrierConfigs = elf.addBinaryDataSection<char>();
    barrierConfigs->setName(".text.BarrierConfigs");
    barrierConfigs->setAddrAlign(64);
    for (int i = 0; i < 7; i++)
        barrierConfigs->addData(i);
    */

    /*
    for (int idx = 0; idx < ELFSectionIndex; idx++) {
        llvm::dbgs() << "Creating section with name " << sectionAttributes[idx].sectionName
                     << " and serializedData.size() = " << sectionAttributes[idx].serializedData.size() << ".\n";

        ELFSection[idx] = elf.addBinaryDataSection<char>();
        ELFSection[idx]->setName(sectionAttributes[idx].sectionName);
        // ELFSection[idx]->set_type(sectionAttributes[idx].sectionType);
        // ELFSection[idx]->setType(sectionAttributes[idx].sectionType); // TODO
        // ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
        ELFSection[idx]->setFlags(sectionAttributes[idx].sectionFlags);
        // ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
        ELFSection[idx]->setAddrAlign(sectionAttributes[idx].sectionAddrAlignInfo);

        // ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
        //                          sectionAttributes[idx].serializedData.size());
        // for (std::size_t i = 0; i < sectionAttributes[idx].serializedData.size(); i++)
        //    ELFSection[idx]->addData(sectionAttributes[idx].serializedData[i]);
        ELFSection[idx]->addData(sectionAttributes[idx].serializedData.data(),
                                 sectionAttributes[idx].serializedData.size());
    }
    */

    // elf::writer::BinaryDataSection<char>* weights_sec = elf.addBinaryDataSection<char>();
    // weights_sec->setName(".data.Weights");
    // weights_sec->set_type(SHT_PROGBITS); // TODO
    // barrierConfigs->setAddrAlign(64);

#ifdef USE_ELFIO
    // Code taken from ELFIO-master/examples/write_obj/write_obj.cpp (and writer/writer.cpp)
    ELFIO::elfio writer;

    // Alex: the following 4 calls don't have to be I guess part of the ELF MLIR
    //    program - these calls establish the main characteristics of the
    //    VPUIP-related ELF happen here in translate.
    // You can't proceed without this function call!
    writer.create(ELFCLASS64, ELFDATA2LSB);

    writer.set_os_abi(ELFOSABI_LINUX);
    writer.set_type(ET_EXEC);
    // writer.set_machine( EM_X86_64 );
    writer.set_machine(EM_res035);

    for (int idx = 0; idx < ELFSectionIndex; idx++) {
        llvm::dbgs() << "Creating section with name " << sectionAttributes[idx].sectionName << ".\n";
        ELFSection[idx] = writer.sections.add(sectionAttributes[idx].sectionName.c_str());
        ELFSection[idx]->set_type(sectionAttributes[idx].sectionType);
        ELFSection[idx]->set_flags(sectionAttributes[idx].sectionFlags);
        ELFSection[idx]->set_addr_align(sectionAttributes[idx].sectionAddrAlignInfo);
        ELFSection[idx]->set_data(sectionAttributes[idx].serializedData.data(),
                                  sectionAttributes[idx].serializedData.size());
    }

#if 0
    // Following the names of sections given by Andrew in elf-blob-example
    // Create DMA section
    // IMPORTANT NOTE: This section is exposed to the ELF dialect
    //   TODO: Use parsed values
    ELFIO::section* dmaTasks_sec = writer.sections.add(".text.dmaTasks");
    dmaTasks_sec->set_type(SHT_PROGBITS);
    // dmaTasks_sec->set_flags(SHF_ALLOC | SHF_EXECINSTR);
    dmaTasks_sec->set_flags(SHF_EXECINSTR);  // Following the "ELF PoC" presentation of Andrew
    dmaTasks_sec->set_addr_align(0x10);
    dmaTasks_sec->set_data(DMATasksELFBLOB.data(), DMATasksELFBLOB.size());

    // IMPORTANT NOTE: This section is exposed to the ELF dialect
    //   TODO: Use parsed values
    ELFIO::section* barriers_sec = writer.sections.add(".text.BarrierConfigs");
    barriers_sec->set_type(SHT_PROGBITS);
    // barriers_sec->set_flags(SHF_ALLOC | SHF_EXECINSTR);
    barriers_sec->set_flags(SHF_EXECINSTR);
    barriers_sec->set_addr_align(0x10);
    barriers_sec->set_data(ConfigureBarriersELFBLOB.data(), ConfigureBarriersELFBLOB.size());
#endif

    ELFIO::section* weights_sec = writer.sections.add(".data.Weights");
    weights_sec->set_type(SHT_PROGBITS);

    // Create string table section
    ELFIO::section* str_sec = writer.sections.add(".strtab");
    str_sec->set_type(SHT_STRTAB);
    //
    // Create string table writer
    ELFIO::string_section_accessor stra(str_sec);
    // Add label name
    ELFIO::Elf32_Word str_index = stra.add_string(".memref.arg0_input");
    //
    ELFIO::section* sym_sec = writer.sections.add(".symtab");
    sym_sec->set_type(SHT_SYMTAB);
    sym_sec->set_info(1);
    sym_sec->set_addr_align(0x4);
    sym_sec->set_entry_size(writer.get_default_entry_size(SHT_SYMTAB));
    sym_sec->set_link(str_sec->get_index());
    //
    // Create symbol table writer
    ELFIO::symbol_section_accessor syma(writer, sym_sec);
    // Add symbol entry (msg has offset == 29)
    ELFIO::Elf_Word sym_to_adjust = syma.add_symbol(str_index, 29, 0, STB_GLOBAL, STT_OBJECT, 0,
                                                    // dmaTasks_sec->get_index()
                                                    ELFSection[0]->get_index());  // TODO: put right section again
    // Another way to add symbol
    // syma.add_symbol(stra, ".memref.arg1_output", 0x00000000, 0, STB_WEAK, STT_FUNC, 0, dmaTasks_sec->get_index());
    syma.add_symbol(stra, ".memref.arg1_output", 0x00000000, 0, STB_GLOBAL, STT_OBJECT, 0,
                    // dmaTasks_sec->get_index()
                    ELFSection[0]->get_index());  // TODO: put right section again

    ELFIO::section* reloctab1_sec = writer.sections.add(".rlt.MappedInference");
    reloctab1_sec->set_type(SHT_REL);
    reloctab1_sec->set_flags(SHF_EXECINSTR);  // MEGA-TODO: use new flag SHF_JIT suggested by Andrew
    reloctab1_sec->set_info(
            // dmaTasks_sec->get_index()
            ELFSection[0]->get_index()  // TODO: put right section again
    );
    reloctab1_sec->set_addr_align(0x4);
    reloctab1_sec->set_entry_size(writer.get_default_entry_size(SHT_REL));
    reloctab1_sec->set_link(sym_sec->get_index());

    ELFIO::section* reloctab2_sec = writer.sections.add(".rlt.jitDMA");
    reloctab2_sec->set_type(SHT_REL);
    reloctab2_sec->set_flags(SHF_EXECINSTR);  // MEGA-TODO: use new flag SHF_JIT suggested by Andrew
    reloctab2_sec->set_info(
            // dmaTasks_sec->get_index()
            ELFSection[0]->get_index()  // TODO: put right section again
    );
    reloctab2_sec->set_addr_align(0x4);
    reloctab2_sec->set_entry_size(writer.get_default_entry_size(SHT_REL));
    reloctab2_sec->set_link(sym_sec->get_index());
    //
    ELFIO::Elf64_Addr place_to_adjust = 11;  // Alex: TODO:change - the offset where we have to patch the address
    //
    // Create relocation table writer
    ELFIO::relocation_section_accessor rela(writer, reloctab2_sec);
    // Add relocation entry (adjust address at offset 11)
    rela.add_entry(place_to_adjust, sym_to_adjust,
                   // R_386_RELATIVE described at https://docs.oracle.com/cd/E19957-01/806-0641/chapter6-26/index.html
                   (unsigned char)R_386_RELATIVE);
    /*
    // Another method to add the same relocation entry at one step is:
    // rela.add_entry( stra, "msg",
    //                 syma, 29, 0,
    //                 ELF_ST_INFO( STB_GLOBAL, STT_OBJECT ), 0,
    //                 text_sec->get_index(),
    //                 place_to_adjust, (unsigned char)R_386_RELATIVE );
    */

    // Create ELF file
    writer.save("vpux_elf_MTL");
#endif  // #ifdef USE_ELFIO

    /*
    //
    // Relocations
    //

    auto dmaTasksRelocation = elf.addRelocationSection();
    dmaTasksRelocation->setSymbolTable(symbolSection);
    // dmaTasksRelocation->setSectionToPatch(dmaTasks);
    dmaTasksRelocation->setSectionToPatch(ELFSection[0]);
    dmaTasksRelocation->setName(".rela.dma");

    auto inputDMA = dmaTasksRelocation->addRelocationEntry();
    inputDMA->setSymbol(inputSym);
    inputDMA->setAddend(0);

    auto outputDMA = dmaTasksRelocation->addRelocationEntry();
    outputDMA->setSymbol(outputSym);
    outputDMA->setAddend(0);
    */
    llvm::dbgs() << "exportVPUIPELF(): ELFSectionIndex = " << ELFSectionIndex << "\n";
    llvm::dbgs().flush();

    elf.write("vpux_elf_MTL");

    llvm::dbgs() << "When exiting exportVPUIPELF(): ELFSectionIndex = " << ELFSectionIndex << "\n";

    //(void)module;
    (void)output;

    return mlir::success();
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        mlir::TranslateToMLIRRegistration("import-IE", importIE);
        mlir::TranslateToMLIRRegistration("import-HWTEST", importHWTEST);
        mlir::TranslateToMLIRRegistration("import-VPUIP", importVPUIP);
        mlir::TranslateFromMLIRRegistration("export-VPUIP", exportVPUIP, registerDialects);
        mlir::TranslateFromMLIRRegistration("export-VPUIP-ELF", exportVPUIPELF, registerDialects);

        return mlir::asMainReturnCode(mlir::mlirTranslateMain(argc, argv, "VPUX Translation Testing Tool"));
    } catch (const std::exception& e) {
        llvm::dbgs() << "Alex: main(): caught exception\n";

        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
