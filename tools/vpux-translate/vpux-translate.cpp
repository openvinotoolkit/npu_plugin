//
// Copyright 2020 Intel Corporation.
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
#include "vpux/compiler/dialect/ELF/ops.hpp"  // 2021_10_07
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"  // Alex
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

#include <elfio/elfio.hpp>       // Alex
#include "llvm/Support/Debug.h"  // Alex

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

// 2021_08_20: Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void printOperation(mlir::Operation* op) {
    // Print the operation itself and some of its properties
    llvm::dbgs() << "visiting op: '" << op->getName() << "' with " << op->getNumOperands() << " operands and "
                 << op->getNumResults() << " results\n";
    fflush(stdout);
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
        llvm::dbgs() << op->getAttrs().size() << " attributes:\n";
        for (mlir::NamedAttribute attr : op->getAttrs())
            llvm::dbgs() << " - '" << attr.first << "' : '" << attr.second << "'\n";
        fflush(stdout);
    }
}

int ELFSectionIndex = 0;  // TODO: change accordingly

std::vector<char> DMATasksELFBLOB;
std::vector<char> ConfigureBarriersELFBLOB;

#define NUM_SECTIONS_MAX 100
struct ELFSectionAttributes {
    int sectionType;
    int sectionFlags;
    int sectionInfo;
    int sectionAddrAlignInfo;
} sectionAttributes[NUM_SECTIONS_MAX];

void printRegion(mlir::Region& region);

// Inspired from https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
void printBlock(mlir::Block& block) {
    // Print the block intrinsics properties (basically: argument list)
    llvm::dbgs() << "Entered printBlock(). Block with " << block.getNumArguments() << " arguments, "
                 << block.getNumSuccessors()
                 << " successors, and "
                 // Note, this `.size()` is traversing a linked-list and is O(n).
                 << block.getOperations().size() << " operations\n";
    fflush(stdout);

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
            llvm::cast<vpux::VPUIPRegMapped::NNDMAOp>(op).serialize(buffer);

            for (long unsigned i = 0; i < buffer.size(); i++) {
                DMATasksELFBLOB.push_back(buffer[i]);
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
            //fflush(stdout);
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

                for (long unsigned i = 0; i < buffer.size(); i++) {
                    ConfigureBarriersELFBLOB.push_back(buffer[i]);
                }
            }
        } else if (opName.find("CreateSection") != std::string::npos) {
            llvm::dbgs() << "exportVPUIPELF(): Found ELF.CreateSectionOp\n";

            llvm::dbgs() << "  exportVPUIPELF(): ELFSectionIndex = " << ELFSectionIndex << "\n";
            ELFSectionIndex++;  // TODO: change accordingly - make nicer (use e.g. ELFIO::section, etc)

            vpux::ELF::CreateSectionOp opSection = llvm::cast<vpux::ELF::CreateSectionOp>(op);

            // Inspired from
            // https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
            sectionAttributes[ELFSectionIndex].sectionType = static_cast<uint32_t>(opSection.secType());
            llvm::dbgs() << "exportVPUIPELF(): secType() = " << sectionAttributes[ELFSectionIndex].sectionType << "\n";

            sectionAttributes[ELFSectionIndex].sectionFlags = static_cast<uint32_t>(opSection.secFlags());
            llvm::dbgs() << "exportVPUIPELF(): secFlags() = " << sectionAttributes[ELFSectionIndex].sectionFlags
                         << "\n";

            sectionAttributes[ELFSectionIndex].sectionInfo = opSection.secInfo();
            llvm::dbgs() << "exportVPUIPELF(): secInfo() = " << sectionAttributes[ELFSectionIndex].sectionInfo << "\n";

            sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo = opSection.secAddrAlign();
            llvm::dbgs() << "exportVPUIPELF(): secAddrAlign() = "
                         << sectionAttributes[ELFSectionIndex].sectionAddrAlignInfo << "\n";

            // See ...ELF/generated/ops.hpp.inc
            // TODO: check that the region contains only the same kind of Op (only NNDMAOp or only ConfigureBarrierOp)
            mlir::Region& aRegion = opSection.aRegion();
            llvm::dbgs() << "exportVPUIPELF(): Calling printRegion(aRegion)\n";
            printRegion(aRegion);
        }

        llvm::dbgs() << "DMATasksELFBLOB.size() = " << DMATasksELFBLOB.size() << "\n";
        llvm::dbgs() << "ConfigureBarriersELFBLOB.size() = " << ConfigureBarriersELFBLOB.size() << "\n";
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

    for (mlir::Operation& op : module) {
        if (mlir::isa<mlir::FuncOp>(op)) {
            auto func = llvm::cast<mlir::FuncOp>(op);  // use maybe mlir::cast
            llvm::dbgs() << "  Function name: " << func.getName() << "\n";

            printRegion(*(func.getCallableRegion()));
        }
    }

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

    ELFIO::section* weights_sec = writer.sections.add(".data.Weights");
    weights_sec->set_type(SHT_PROGBITS);

    // IMPORTANT NOTE: This section is exposed to the ELF dialect
    //   TODO: Use parsed values
    ELFIO::section* barriers_sec = writer.sections.add(".text.BarrierConfigs");
    barriers_sec->set_type(SHT_PROGBITS);
    // barriers_sec->set_flags(SHF_ALLOC | SHF_EXECINSTR);
    barriers_sec->set_flags(SHF_EXECINSTR);
    barriers_sec->set_addr_align(0x10);
    barriers_sec->set_data(ConfigureBarriersELFBLOB.data(), ConfigureBarriersELFBLOB.size());

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
    ELFIO::Elf_Word sym_to_adjust =
            syma.add_symbol(str_index, 29, 0, STB_GLOBAL, STT_OBJECT, 0, dmaTasks_sec->get_index());
    // Another way to add symbol
    // syma.add_symbol(stra, ".memref.arg1_output", 0x00000000, 0, STB_WEAK, STT_FUNC, 0, dmaTasks_sec->get_index());
    syma.add_symbol(stra, ".memref.arg1_output", 0x00000000, 0, STB_GLOBAL, STT_OBJECT, 0, dmaTasks_sec->get_index());

    ELFIO::section* reloctab1_sec = writer.sections.add(".rlt.MappedInference");
    reloctab1_sec->set_type(SHT_REL);
    reloctab1_sec->set_flags(SHF_EXECINSTR);  // MEGA-TODO: use new flag SHF_JIT suggested by Andrew
    reloctab1_sec->set_info(dmaTasks_sec->get_index());
    reloctab1_sec->set_addr_align(0x4);
    reloctab1_sec->set_entry_size(writer.get_default_entry_size(SHT_REL));
    reloctab1_sec->set_link(sym_sec->get_index());

    ELFIO::section* reloctab2_sec = writer.sections.add(".rlt.jitDMA");
    reloctab2_sec->set_type(SHT_REL);
    reloctab2_sec->set_flags(SHF_EXECINSTR);  // MEGA-TODO: use new flag SHF_JIT suggested by Andrew
    reloctab2_sec->set_info(dmaTasks_sec->get_index());
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
