//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <kernels/inc/common_types.h>

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Convert2VPUIPRegMappedAndELFPass
//

class Convert2VPUIPRegMappedAndELFPass final :
        public Convert2VPUIPRegMappedAndELFBase<Convert2VPUIPRegMappedAndELFPass> {
public:
    explicit Convert2VPUIPRegMappedAndELFPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename DerivedOpType, typename CreateSectionOpType>
    mlir::Value createSection(mlir::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr,
                              vpux::ELF::SectionTypeAttr secType, vpux::ELF::SectionFlagsAttr secFlags);

    mlir::Value createCMXMappingSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx);
    mlir::Value lookupELFSymbol(mlir::Value& symtabValue, mlir::Value& sym_input_value);
    mlir::Value createBuffersSecAndSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx);
    void createNetworkIOSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx, vpux::IE::CNNNetworkOp cnnOp);
    void createDMARelocs(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& dmaSectionValue);
    void createKernelParamsRelocs(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& kernelParamsSectionValue);
    void createActKernelRelocs(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& actKernelRangeSectionValue,
                               mlir::Value& actKernelInvoSectionValue);
    void setupActKernelRtConfigs(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& mappedInferenceSectionOpValue);

    void safeRunOnModule() final;

private:
    Logger _log;

    mlir::Value networkInputSymTabValue, networkOutputSymTabValue;

    mlir::Value tasksSymTabValue, bufferSymTabValue, CMXMappingSymtabValue;

    std::map<std::string, mlir::Value> symbolMap;

    // map that correlates between Const::DeclareOp values and their ELF::SymbolOp value
    llvm::MapVector<mlir::Value, mlir::Value> constSymMap;

    // map that correlates between Const::DeclareOp values and their offset in the .data.const section
    llvm::MapVector<mlir::Value, size_t> constOffsetMap;

    std::vector<ELF::SymbolOp> elfCMXMappingSyms;

    // task counts
    unsigned int dmaCount = 0;
    unsigned int barrierCount = 0;
    unsigned int rangeCount = 0;
    unsigned int invoCount = 0;
};

// createSection() creates an ELF::CreateSectionOp and puts into its body
//   an ELF.PutOpInSectionOp instruction for each object of type DerivedOpType
//   from func (a FuncOp).
template <typename DerivedOpType, typename CreateSectionOpType>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                            std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                            vpux::ELF::SectionFlagsAttr secFlags) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    CreateSectionOpType elfCreateSectionOp =
            builderFunc.create<CreateSectionOpType>(mlir::UnknownLoc::get(ctx),
                                                    sectionType,               // mlir::Type
                                                    secNameStr,                // llvm::StringRef secName,
                                                    secType,                   // vpux::ELF::SectionTypeAttr secType,
                                                    secFlags,                  // vpux::ELF::SectionFlagsAttr secFlags,
                                                    elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                    elf::VPU_SH_ADDR_ALIGN_FOR_VPU  // int64_t secAddrAlign
            );

    auto* elfCreateSectionOperation = elfCreateSectionOp.getOperation();

    mlir::Block* blkNew = &(elfCreateSectionOperation->getRegion(0).emplaceBlock());

    for (DerivedOpType op : func.getOps<DerivedOpType>()) {
        if (auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(&op)) {
            if (declareBufferOp->section() != vpux::VPURT::BufferSection::DDR) {
                continue;
            }
        }

        mlir::OpBuilder builderElfSectionOpReg(blkNew, blkNew->end());

        builderElfSectionOpReg.create<ELF::PutOpInSectionOp>(
                builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                op.getResult()                           // mlir::Value inputArg
        );
    }

    return elfCreateSectionOperation->getResult(0);
}

template <>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection<Const::DeclareOp, ELF::CreateSectionOp>(
        mlir::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
        vpux::ELF::SectionFlagsAttr secFlags) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    auto elfCreateSectionOp =
            builderFunc.create<ELF::CreateSectionOp>(mlir::UnknownLoc::get(ctx),
                                                     sectionType,               // mlir::Type
                                                     secNameStr,                // llvm::StringRef secName,
                                                     secType,                   // vpux::ELF::SectionTypeAttr secType,
                                                     secFlags,                  // vpux::ELF::SectionFlagsAttr secFlags,
                                                     elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                     elf::VPU_SH_ADDR_ALIGN_FOR_VPU  // int64_t secAddrAlign
            );

    auto* elfCreateSectionOperation = elfCreateSectionOp.getOperation();

    mlir::Block* blkSectionOp = &(elfCreateSectionOperation->getRegion(0).emplaceBlock());

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    size_t constCount = 0;
    size_t totalOffset = 0;

    for (Const::DeclareOp op : func.getOps<Const::DeclareOp>()) {
        std::string symName = "sym_const" + std::to_string(constCount);
        constCount++;

        mlir::Value constOpValue = op.getResult();

        auto elfSymbolOp = builderFunc.create<ELF::SymbolOp>(builderFunc.getUnknownLoc(),
                                                             vpux::ELF::SymbolType::get(ctx),  // mlir::Type
                                                             constOpValue,                     // mlir::Value inputArg
                                                             isBuiltin,                        // mlir::UnitAttr
                                                             mlir::StringAttr::get(ctx, symName),  // mlir::StringAttr
                                                             typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                                             sizeSym,  // size
                                                             valueSym  // value
        );

        mlir::Value constSymValue = elfSymbolOp.getResult();

        constSymMap[constOpValue] = constSymValue;
        constOffsetMap[constOpValue] = totalOffset;

        mlir::OpBuilder builderElfSectionOp(blkSectionOp, blkSectionOp->end());

        builderElfSectionOp.create<ELF::PutOpInSectionOp>(builderElfSectionOp.getUnknownLoc(),  // endOp->getLoc(),
                                                          op.getResult()                        // mlir::Value inputArg
        );

        totalOffset += op.getBinarySize();
    }

    return elfCreateSectionOperation->getResult(0);
}

mlir::Value Convert2VPUIPRegMappedAndELFPass::createBuffersSecAndSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx) {
    mlir::Value bufferSectionOpValue = createSection<vpux::VPURT::DeclareBufferOp, ELF::CreateLogicalSectionOp>(
            func, ctx, ".data.BuffersIO", vpux::ELF::SectionTypeAttr::SHT_NOBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC);

    createSection<vpux::Const::DeclareOp, ELF::CreateSectionOp>(func, ctx, ".data.ConstIO",
                                                                vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                                vpux::ELF::SectionFlagsAttr::SHF_ALLOC);

    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    auto bufferSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),                  // mlir::Type
                                              bufferSectionOpValue,                             // mlir::Value inputArg
                                              isBuiltin,                                        // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_bufferSection"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_bufferSection"] = bufferSectionSym.getResult();

    auto bufferSymTabOp =
            builderFunc.create<ELF::CreateSymbolTableSectionOp>(mlir::UnknownLoc::get(ctx),
                                                                vpux::ELF::SectionType::get(ctx),  // mlir::Type
                                                                ".symtab.buffers",  // llvm::StringRef secName,
                                                                vpux::ELF::SectionFlagsAttr::SHF_NONE,
                                                                isBuiltin  // mlir::UnitAttr
            );

    mlir::Region& regBufferSymTabOp = bufferSymTabOp.getOperation()->getRegion(0);
    mlir::Block* blkBufferSymTabOp = new mlir::Block();
    regBufferSymTabOp.push_back(blkBufferSymTabOp);
    mlir::OpBuilder builderBufferSymTab(blkBufferSymTabOp, blkBufferSymTabOp->begin());

    mlir::Value bufferSectionSymValue = bufferSectionSym.getResult();
    mlir::Value bufferSymTabValue = bufferSymTabOp.getResult();

    builderBufferSymTab.create<ELF::PutOpInSectionOp>(builderBufferSymTab.getUnknownLoc(), bufferSectionSymValue);

    for (auto constSymValuePair : constSymMap) {
        builderBufferSymTab.create<ELF::PutOpInSectionOp>(builderBufferSymTab.getUnknownLoc(),
                                                          constSymValuePair.second);
    }

    return bufferSymTabValue;
}

mlir::Value Convert2VPUIPRegMappedAndELFPass::createCMXMappingSymtab(mlir::FuncOp funcOp, mlir::MLIRContext* ctx) {
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    std::vector<mlir::arith::ConstantOp> symVals;

    vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = mlir::UnitAttr::get(ctx);

    for (unsigned i = 0; i <= vpux::ELF::getMaxEnumValForCMXMappingSymbolAttr(); ++i) {
        auto optionalCMXMappingSymValue = vpux::ELF::symbolizeCMXMappingSymbolAttr(i);
        if (!optionalCMXMappingSymValue.hasValue())
            continue;

        auto CMXMappingSymValue = optionalCMXMappingSymValue.getValue();
        auto CMXMappingSymStringRef = vpux::ELF::stringifyCMXMappingSymbolAttr(CMXMappingSymValue);

        symVals.push_back(builderFunc.create<mlir::arith::ConstantIntOp>(mlir::UnknownLoc::get(ctx), i, 8));
        elfCMXMappingSyms.push_back(builderFunc.create<ELF::SymbolOp>(
                mlir::UnknownLoc::get(ctx),
                symbolType,                                                            // mlir::Type
                symVals[i],                                                            // mlir::Value inputArg
                isBuiltin, mlir::StringAttr::get(ctx, CMXMappingSymStringRef.data()),  // mlir::StringAttr
                typeSym,                                                               // vpux::ELF::SymbolTypeAttrAttr
                sizeSym,                                                               // size
                valueSym                                                               // value
                ));
    }

    vpux::ELF::SectionType secType = vpux::ELF::SectionType::get(ctx);

    ELF::CreateSymbolTableSectionOp createCMXMappingSymtabOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            secType,                                // mlir::Type
            "VPU_RT_SYMTAB",                        // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,  // vpux::ELF::SectionFlagsAttr secFlags,
            isBuiltin);

    mlir::Region& regCMXMappingSymtab = createCMXMappingSymtabOp.getOperation()->getRegion(0);
    mlir::Block* blkCMXMappingSymtab = new mlir::Block();

    regCMXMappingSymtab.push_back(blkCMXMappingSymtab);

    mlir::OpBuilder builderCMXMappingSymtab(blkCMXMappingSymtab, blkCMXMappingSymtab->begin());

    for (auto elfCMXMappingSym : elfCMXMappingSyms) {
        builderCMXMappingSymtab.create<ELF::PutOpInSectionOp>(
                builderCMXMappingSymtab.getUnknownLoc(),  // endOp->getLoc(),
                elfCMXMappingSym.getResult()              // mlir::Value inputArg
        );
    }

    return createCMXMappingSymtabOp.getResult();
}

void Convert2VPUIPRegMappedAndELFPass::createNetworkIOSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                             vpux::IE::CNNNetworkOp cnnOp) {
    SmallVector<vpux::IE::DataInfoOp, 1> dataInfoOpInVec = cnnOp.getInputsInfo();
    SmallVector<vpux::IE::DataInfoOp, 1> dataInfoOpOutVec = cnnOp.getOutputsInfo();

    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    std::vector<mlir::Value> inputSyms;
    std::vector<mlir::Value> outputSyms;

    mlir::IntegerType uint64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    for (auto funcArg : func.getArguments()) {
        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        vpux::ELF::SymbolTypeAttrAttr typeSym;
        mlir::IntegerAttr valueSym;
        mlir::UnitAttr isBuiltin = nullptr;

        auto argMemrefType = funcArg.getType().cast<mlir::MemRefType>();
        mlir::IntegerAttr sizeSym = mlir::IntegerAttr::get(uint64Type, argMemrefType.getSizeInBits() / 8);

        bool isInputSym = funcArg.getArgNumber() < dataInfoOpInVec.size();

        auto index = isInputSym ? funcArg.getArgNumber() : funcArg.getArgNumber() - dataInfoOpInVec.size();
        mlir::StringAttr nameSym = isInputSym ? mlir::StringAttr::get(ctx, dataInfoOpInVec[index].name())
                                              : mlir::StringAttr::get(ctx, dataInfoOpOutVec[index].name());

        auto netIOSym = builderFunc.create<ELF::SymbolOp>(builderFunc.getUnknownLoc(),
                                                          symbolType,  // mlir::Type
                                                          funcArg,     // mlir::Value inputArg
                                                          isBuiltin,   // mlir::UnitAttr
                                                          nameSym,     // mlir::StringAttr
                                                          typeSym,     // vpux::ELF::SymbolTypeAttrAttr
                                                          sizeSym,     // size
                                                          valueSym     // value
        );

        if (isInputSym) {
            inputSyms.push_back(netIOSym.getResult());
        } else {
            outputSyms.push_back(netIOSym.getResult());
        }
    }

    // Secondly we create the symbol table for the input symbols
    ELF::CreateSymbolTableSectionOp createInputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SectionType::get(ctx),                // mlir::Type
            ".symtab.input",                                 // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT,  // vpux::ELF::SectionFlagsAttr secFlags,
            nullptr                                          // mlir::UnitAttr
    );
    //
    mlir::Region& regInputSymTabSec = createInputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regInputSymTabSec.push_back(blkInputSymTabSec);
    //
    mlir::OpBuilder builderInputSymTabSec(blkInputSymTabSec, blkInputSymTabSec->begin());
    networkInputSymTabValue = createInputSymTableSectionOp.getResult();

    // Thirdly we create the symbol table for the output symbols
    ELF::CreateSymbolTableSectionOp createOutputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SectionType::get(ctx),                 // mlir::Type
            ".symtab.output",                                 // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT,  // vpux::ELF::SectionFlagsAttr secFlags,
            nullptr                                           // mlir::UnitAttr
    );
    //
    mlir::Region& regOutputSymTabSec = createOutputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regOutputSymTabSec.push_back(blkOutputSymTabSec);
    //
    mlir::OpBuilder builderOutputSymTabSec(blkOutputSymTabSec, blkOutputSymTabSec->begin());
    networkOutputSymTabValue = createOutputSymTableSectionOp.getResult();

    for (auto inputSym : inputSyms) {
        builderInputSymTabSec.create<ELF::PutOpInSectionOp>(builderInputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                                                            inputSym  // mlir::Value inputArg
        );
    }

    for (auto outputSym : outputSyms) {
        builderOutputSymTabSec.create<ELF::PutOpInSectionOp>(
                builderOutputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                outputSym                                // mlir::Value inputArg
        );
    }
}

void Convert2VPUIPRegMappedAndELFPass::createKernelParamsRelocs(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                                mlir::Value& kernelParamsSectionValue) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELF::CreateRelocationSectionOp createKernelParamsRelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.KernelParams",                        // llvm::StringRef secName,
                    tasksSymTabValue,                           // sourceSymbolTableSection,
                    kernelParamsSectionValue,                   // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regKernelParamsRelocSec = createKernelParamsRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkKernelParamsRelocSec = new mlir::Block();
    regKernelParamsRelocSec.push_back(blkKernelParamsRelocSec);

    mlir::OpBuilder builderKernelParamsRelocSec(blkKernelParamsRelocSec, blkKernelParamsRelocSec->begin());

    ELF::CreateRelocationSectionOp createKernelParamsIORelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.KernelParamsIO",                      // llvm::StringRef secName,
                    bufferSymTabValue,                          // sourceSymbolTableSection,
                    kernelParamsSectionValue,                   // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regKernelParamsIORelocSec = createKernelParamsIORelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkKernelParamsIORelocSec = new mlir::Block();
    regKernelParamsIORelocSec.push_back(blkKernelParamsIORelocSec);

    mlir::OpBuilder builderKernelParamsIORelocSec(blkKernelParamsIORelocSec, blkKernelParamsIORelocSec->begin());

    ELF::CreateRelocationSectionOp createKernelParamsCMXIORelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.KernelParamsIO_CMX",                  // llvm::StringRef secName,
                    CMXMappingSymtabValue,                      // sourceSymbolTableSection,
                    kernelParamsSectionValue,                   // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regKernelParamsCMXIORelocSec =
            createKernelParamsCMXIORelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkKernelParamsCMXIORelocSec = new mlir::Block();
    regKernelParamsCMXIORelocSec.push_back(blkKernelParamsCMXIORelocSec);

    mlir::OpBuilder builderKernelParamsCMXIORelocSec(blkKernelParamsCMXIORelocSec,
                                                     blkKernelParamsCMXIORelocSec->begin());

    mlir::Value kernelParamsSectionSym = symbolMap["sym_kernelParamsSection"];
    mlir::Value bufferSectionSymValue = symbolMap["sym_bufferSection"];

    auto kernelParamsOps = func.getOps<vpux::VPUIPRegMapped::KernelParamsOp>();

    size_t total_addend = 0;

    for (auto kernelParamsOp : kernelParamsOps) {
        auto partial_addend = total_addend + kernelParamsOp.getParamsStructSize();

        // input addr
        if (auto kernelInputOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(kernelParamsOp.input().getDefiningOp())) {
            if (kernelInputOp.section() == VPURT::BufferSection::DDR) {
                builderKernelParamsIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderKernelParamsIORelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                        offsetof(sw_params::MemRefData, dataAddr), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                        bufferSectionSymValue, kernelInputOp.byteOffset());
            } else if (kernelInputOp.section() == VPURT::BufferSection::CMX_NN) {
                builderKernelParamsCMXIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderKernelParamsCMXIORelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                        offsetof(sw_params::MemRefData, dataAddr), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                        elfCMXMappingSyms[static_cast<int>(
                                                  vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                .getResult(),
                        kernelInputOp.byteOffset());
            }
        } else if (auto kernelConstInputOp = mlir::dyn_cast<Const::DeclareOp>(kernelParamsOp.input().getDefiningOp())) {
            builderKernelParamsIORelocSec.create<ELF::RelocImmOffsetOp>(
                    builderKernelParamsIORelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                    offsetof(sw_params::MemRefData, dataAddr), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                    constSymMap[kernelConstInputOp.getResult()], constOffsetMap[kernelConstInputOp.getResult()]);
        } else {
            VPUX_THROW("Unsupported ActShave task input");
        }

        // output addr
        if (auto kernelOutputOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(kernelParamsOp.output().getDefiningOp())) {
            if (kernelOutputOp.section() == VPURT::BufferSection::DDR) {
                builderKernelParamsIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderKernelParamsIORelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                        sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dataAddr),
                        vpux::ELF::RelocationTypeAttr::R_VPU_32, bufferSectionSymValue, kernelOutputOp.byteOffset());
            } else if (kernelOutputOp.section() == VPURT::BufferSection::CMX_NN) {
                builderKernelParamsCMXIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderKernelParamsCMXIORelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                        sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dataAddr),
                        vpux::ELF::RelocationTypeAttr::R_VPU_32,
                        elfCMXMappingSyms[static_cast<int>(
                                                  vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                .getResult(),
                        kernelOutputOp.byteOffset());
            }
        } else {
            VPUX_THROW("Unsupported ActShave task output");
        }

        // input Dims addr
        builderKernelParamsRelocSec.create<ELF::RelocImmOffsetOp>(
                builderKernelParamsRelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                offsetof(sw_params::MemRefData, dimsAddr), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                kernelParamsSectionSym, partial_addend);

        partial_addend += sizeof(int32_t) * getShape(kernelParamsOp.input()).size();

        // output Dims addr
        builderKernelParamsRelocSec.create<ELF::RelocImmOffsetOp>(
                builderKernelParamsRelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dimsAddr),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);

        partial_addend += sizeof(int32_t) * getShape(kernelParamsOp.output()).size();

        // input Strides addr
        builderKernelParamsRelocSec.create<ELF::RelocImmOffsetOp>(
                builderKernelParamsRelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                offsetof(sw_params::MemRefData, stridesAddr), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                kernelParamsSectionSym, partial_addend);

        partial_addend += sizeof(int64_t) * getMemStrides(kernelParamsOp.output()).size();

        // input Strides addr
        builderKernelParamsRelocSec.create<ELF::RelocImmOffsetOp>(
                builderKernelParamsRelocSec.getUnknownLoc(), kernelParamsOp.getResult(),
                sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, stridesAddr),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);

        total_addend += kernelParamsOp.getBinarySize();
    }
}

mlir::Value Convert2VPUIPRegMappedAndELFPass::lookupELFSymbol(mlir::Value& symtabValue, mlir::Value& sym_input_value) {
    auto symtabOp = llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(symtabValue.getDefiningOp());

    auto symtabBlk = symtabOp.getBody();
    for (auto& op : symtabBlk->getOperations()) {
        if (auto symOp = llvm::dyn_cast<vpux::ELF::SymbolOp>(op)) {
            if (symOp.inputArg() == sym_input_value) {
                return symOp.getResult();
            }
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELF::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();
            auto symOp = llvm::dyn_cast<vpux::ELF::SymbolOp>(actualOp);
            if (symOp.inputArg() == sym_input_value) {
                return symOp.getResult();
            }
        }
    }

    mlir::Value no_val;
    return no_val;
}

void Convert2VPUIPRegMappedAndELFPass::createActKernelRelocs(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                             mlir::Value& actKernelRangeSectionValue,
                                                             mlir::Value& actKernelInvoSectionValue) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    // range relocs

    ELF::CreateRelocationSectionOp createActKernelRangeRelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.ActKernelRange",                      // llvm::StringRef secName,
                    tasksSymTabValue,                           // sourceSymbolTableSection,
                    actKernelRangeSectionValue,                 // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regActKernelRangeRelocSec = createActKernelRangeRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkActKernelRangeRelocSec = new mlir::Block();
    regActKernelRangeRelocSec.push_back(blkActKernelRangeRelocSec);

    mlir::OpBuilder builderActKernelRangeRelocSec(blkActKernelRangeRelocSec, blkActKernelRangeRelocSec->begin());

    auto actKernelRangeOps = func.getOps<vpux::VPUIPRegMapped::ActKernelRangeOp>();

    size_t total_addend = 0;

    mlir::Value kernelTextSecSym = symbolMap["sym_kernelTextSection"];

    // range text section reloc (textWindowBase_)
    for (auto actKernelRangeOp : actKernelRangeOps) {
        builderActKernelRangeRelocSec.create<ELF::RelocImmOffsetOp>(
                builderActKernelRangeRelocSec.getUnknownLoc(), actKernelRangeOp.getResult(),
                offsetof(host_parsing::ActKernelRangeWrapper, kRange_) +
                        offsetof(host_parsing::ActKernelRange, textWindowBase_),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelTextSecSym, total_addend);

        auto declareKernelTextOp = llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareKernelTextOp>(
                actKernelRangeOp.kernel_text_index().getDefiningOp());
        total_addend += declareKernelTextOp.getBinarySize();
    }

    // invo relocs

    ELF::CreateRelocationSectionOp createActKernelInvoRelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.ActKernelInvo",                       // llvm::StringRef secName,
                    tasksSymTabValue,                           // sourceSymbolTableSection,
                    actKernelInvoSectionValue,                  // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regActKernelInvoRelocSec = createActKernelInvoRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkActKernelInvoRelocSec = new mlir::Block();
    regActKernelInvoRelocSec.push_back(blkActKernelInvoRelocSec);

    mlir::OpBuilder builderActKernelInvoRelocSec(blkActKernelInvoRelocSec, blkActKernelInvoRelocSec->begin());

    auto actKernelInvoOps = func.getOps<vpux::VPUIPRegMapped::ActKernelInvocationOp>();

    size_t dataSec_total_addend = 0;
    size_t paramsSec_total_addend = 0;
    size_t rangeSec_total_addend = 0;

    mlir::Value kernelDataSecSym = symbolMap["sym_kernelDataSection"];
    mlir::Value kernelParamsSecSym = symbolMap["sym_kernelParamsSection"];
    mlir::Value actKernelRangeSecSym = symbolMap["sym_actKernelRangeSection"];

    for (auto actKernelInvoOp : actKernelInvoOps) {
        auto actKernelInvoOpIndex = actKernelInvoOp.index().getType().cast<vpux::VPUIPRegMapped::IndexType>();
        auto associatedRangeOp =
                llvm::dyn_cast<vpux::VPUIPRegMapped::ActKernelRangeOp>(actKernelInvoOp.range_index().getDefiningOp());
        // range reloc
        builderActKernelInvoRelocSec.create<ELF::RelocImmOffsetOp>(
                builderActKernelInvoRelocSec.getUnknownLoc(), actKernelInvoOp.getResult(),
                offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) +
                        offsetof(host_parsing::ActKernelInvocation, range_),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, actKernelRangeSecSym, rangeSec_total_addend);

        rangeSec_total_addend += associatedRangeOp.getBinarySize();

        // data section reloc
        builderActKernelInvoRelocSec.create<ELF::RelocImmOffsetOp>(
                builderActKernelInvoRelocSec.getUnknownLoc(), actKernelInvoOp.getResult(),
                offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) +
                        offsetof(host_parsing::ActKernelInvocation, dataWindowBase_),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelDataSecSym, dataSec_total_addend);

        auto declareKernelArgsOp = llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareKernelArgsOp>(
                associatedRangeOp.kernel_args_index().getDefiningOp());
        dataSec_total_addend += declareKernelArgsOp.getBinarySize();

        // params reloc
        builderActKernelInvoRelocSec.create<ELF::RelocImmOffsetOp>(
                builderActKernelInvoRelocSec.getUnknownLoc(), actKernelInvoOp.getResult(),
                offsetof(host_parsing::ActKernelInvocationWrapper, kInvo_) +
                        offsetof(host_parsing::ActKernelInvocation, kernelArgs_),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSecSym, paramsSec_total_addend);

        auto kernelParamsOps = func.getOps<vpux::VPUIPRegMapped::KernelParamsOp>();
        for (auto kernelParamsOp : kernelParamsOps) {
            auto kernelParamsOpIndex = kernelParamsOp.index().getType().cast<vpux::VPUIPRegMapped::IndexType>();
            if (kernelParamsOpIndex.getValue() == actKernelInvoOpIndex.getValue()) {
                paramsSec_total_addend += kernelParamsOp.getBinarySize();
            }
        }
    }
}

void Convert2VPUIPRegMappedAndELFPass::setupActKernelRtConfigs(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                               mlir::Value& mappedInferenceSectionOpValue) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    const auto bufferMemrefShape = SmallVector<int64_t>{262144};
    auto DDRNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    const auto DDRSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, DDRNameAttr);

    unsigned int perm[1] = {0};

    auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), ctx);

    auto memrefType = mlir::MemRefType::get(bufferMemrefShape, mlir::IntegerType::get(ctx, 32), map, DDRSymbolAttr);

    auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                      memrefType,                 // Type
                                                                      VPURT::BufferSection::DDR,  // Buffer Type
                                                                      64                          // byteOffset
    );

    auto actKRtConfigSecOp = builderFunc.create<ELF::CreateLogicalSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),        // mlir::Type
            ".bss.actKernelRtConfigSec",             // llvm::StringRef secName,
            vpux::ELF::SectionTypeAttr::SHT_NOBITS,  // vpux::ELF::SectionTypeAttr secType,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,   // vpux::ELF::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                // int64_t secInfo,
            1024                                     // int64_t secAddrAlign
    );

    mlir::Block* blkNew = &(actKRtConfigSecOp.declaredOps().emplaceBlock());

    mlir::OpBuilder builderElfSectionOpReg(blkNew, blkNew->end());

    builderElfSectionOpReg.create<ELF::PutOpInSectionOp>(builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                                                         declareBufferOp.getResult()  // mlir::Value inputArg
    );

    mlir::Value actKRtConfigSecValue = actKRtConfigSecOp.getResult();

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    auto actKRtConfigSym = builderFunc.create<ELF::SymbolOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SymbolType::get(ctx),                          // mlir::Type
            actKRtConfigSecValue,                                     // mlir::Value inputArg
            isBuiltin,                                                // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_actKernelRtConfigsSec"),  // mlir::StringAttr
            typeSym,                                                  // vpux::ELF::SymbolTypeAttrAttr
            sizeSym,                                                  // size
            valueSym                                                  // value
    );
    symbolMap["sym_actKernelRtConfigsSec"] = actKRtConfigSym.getResult();

    auto actKRtConfigSymTab = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),  // mlir::Type
            ".symtab.actKernelRtConfig",       // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,
            isBuiltin  // mlir::UnitAttr
    );

    mlir::Region& regActKRtConfigSymTabOp = actKRtConfigSymTab.getOperation()->getRegion(0);
    mlir::Block* blkActKRtConfigSymTabOp = new mlir::Block();
    regActKRtConfigSymTabOp.push_back(blkActKRtConfigSymTabOp);
    mlir::OpBuilder builderActKRtConfigSymTab(blkActKRtConfigSymTabOp, blkActKRtConfigSymTabOp->begin());

    builderActKRtConfigSymTab.create<ELF::PutOpInSectionOp>(builderActKRtConfigSymTab.getUnknownLoc(),
                                                            actKRtConfigSym.getResult());

    mlir::Value actKRtConfigSymValue = actKRtConfigSym.getResult();

    ELF::CreateRelocationSectionOp createMIActKRtConfigsRelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    mlir::UnknownLoc::get(ctx),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.MI_AKRtConfig",                       // llvm::StringRef secName,
                    actKRtConfigSymTab.getResult(),             // sourceSymbolTableSection,
                    mappedInferenceSectionOpValue,              // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regMappedInfRelocSec = createMIActKRtConfigsRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkMappedInfRelocSec = new mlir::Block();
    regMappedInfRelocSec.push_back(blkMappedInfRelocSec);

    mlir::OpBuilder builderMappedInfRelocSec(blkMappedInfRelocSec, blkMappedInfRelocSec->begin());

    for (auto mappedInferenceOp : func.getOps<VPUIPRegMapped::MappedInferenceOp>()) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(host_parsing::MappedInference, actRtConfigs) +
                        offsetof(host_parsing::ActKernelRuntimeConfigs, actRtWindowBase_),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, actKRtConfigSymValue, 0);
    }
}

void Convert2VPUIPRegMappedAndELFPass::createDMARelocs(mlir::FuncOp funcOp, mlir::MLIRContext* ctx,
                                                       mlir::Value& dmaSectionValue) {
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    ELF::CreateRelocationSectionOp createInputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SectionType::get(ctx),  // mlir::Type
            ".rlt.DMA_NetInput",               // llvm::StringRef secName,
            networkInputSymTabValue,           // sourceSymbolTableSection,
            dmaSectionValue,                   // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );

    mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputRelocSec = new mlir::Block();

    regInputRelocSec.push_back(blkInputRelocSec);

    mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

    ELF::CreateRelocationSectionOp createOutputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SectionType::get(ctx),  // mlir::Type
            ".rlt.DMA_NetOutput",              // llvm::StringRef secName,
            networkOutputSymTabValue,          // sourceSymbolTableSection,
            dmaSectionValue,                   // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputRelocSec = new mlir::Block();
    regOutputRelocSec.push_back(blkOutputRelocSec);

    mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

    ELF::CreateRelocationSectionOp createDMAIORelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),           // mlir::Type
            ".rlt.dmaIO",                               // llvm::StringRef secName,
            bufferSymTabValue,                          // sourceSymbolTableSection,
            dmaSectionValue,                            // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
    );

    mlir::Region& regDMAIORelocSec = createDMAIORelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkDMAIORelocSec = new mlir::Block();
    regDMAIORelocSec.push_back(blkDMAIORelocSec);

    mlir::OpBuilder builderDMAIORelocSec(blkDMAIORelocSec, blkDMAIORelocSec->begin());

    ELF::CreateRelocationSectionOp createDMACMXIORelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    builderFunc.getUnknownLoc(),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.dmaIO_CMX",                           // llvm::StringRef secName,
                    CMXMappingSymtabValue,                      // sourceSymbolTableSection,
                    dmaSectionValue,                            // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regDMACMXIORelocSec = createDMACMXIORelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkDMACMXIORelocSec = new mlir::Block();
    regDMACMXIORelocSec.push_back(blkDMACMXIORelocSec);

    mlir::OpBuilder builderDMACMXIORelocSec(blkDMACMXIORelocSec, blkDMACMXIORelocSec->begin());

    mlir::Value bufferSectionSymValue = symbolMap["sym_bufferSection"];

    auto dmaOps = funcOp.getOps<vpux::VPUIPRegMapped::NNDMAOp>();

    for (auto dmaOp : dmaOps) {
        // input addr
        if (auto dmaInputArg = dmaOp.input().dyn_cast<mlir::BlockArgument>()) {
            if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaInputArg)) {
                builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netInputSymValue,                         // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            } else if (mlir::Value netInputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaInputArg)) {
                builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netInputSymValue,                         // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            }
        } else if (auto dmaInputOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.input().getDefiningOp())) {
            if (dmaInputOp.section() == VPURT::BufferSection::DDR) {
                builderDMAIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderDMAIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64, bufferSectionSymValue, dmaInputOp.byteOffset());
            } else if (dmaInputOp.section() == VPURT::BufferSection::CMX_NN) {
                builderDMACMXIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderDMACMXIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,
                        elfCMXMappingSyms[static_cast<int>(
                                                  vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                .getResult(),
                        dmaInputOp.byteOffset());
            }
        } else if (auto dmaConstInputOp = mlir::dyn_cast<Const::DeclareOp>(dmaOp.input().getDefiningOp())) {
            builderDMAIORelocSec.create<ELF::RelocImmOffsetOp>(
                    builderDMAIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                    offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, src),
                    vpux::ELF::RelocationTypeAttr::R_VPU_64, constSymMap[dmaConstInputOp.getResult()],
                    constOffsetMap[dmaConstInputOp.getResult()]);
        } else {
            VPUX_THROW("Unsupported DMA task input");
        }
        // output addr
        if (auto dmaOutputArg = dmaOp.output_buff().dyn_cast<mlir::BlockArgument>()) {
            if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaOutputArg)) {
                builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netOutputSymValue,                        // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            } else if (mlir::Value netOutputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaOutputArg)) {
                builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netOutputSymValue,                        // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            }
        } else if (auto dmaOutputOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.output_buff().getDefiningOp())) {
            if (dmaOutputOp.section() == VPURT::BufferSection::DDR) {
                builderDMAIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderDMAIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64, bufferSectionSymValue, dmaOutputOp.byteOffset());
            } else if (dmaOutputOp.section() == VPURT::BufferSection::CMX_NN) {
                builderDMACMXIORelocSec.create<ELF::RelocImmOffsetOp>(
                        builderDMACMXIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(host_parsing::DmaWrapper, transaction) + offsetof(host_parsing::DmaDescriptor, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,
                        elfCMXMappingSyms[static_cast<int>(
                                                  vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                .getResult(),
                        dmaOutputOp.byteOffset());
            }
        } else {
            VPUX_THROW("Unsupported DMA task output");
        }
        // link_address
        if (dmaCount > dmaOp.getType().getValue() + 1) {
            builderDMACMXIORelocSec.create<ELF::RelocImmOffsetOp>(
                    builderDMACMXIORelocSec.getUnknownLoc(), dmaOp.getResult(),
                    offsetof(host_parsing::DmaWrapper, transaction), vpux::ELF::RelocationTypeAttr::R_VPU_32_RTM,
                    elfCMXMappingSyms[static_cast<int>(vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_DMA0)]
                            .getResult(),
                    sizeof(host_parsing::DmaWrapper));
        }
    }
}

void Convert2VPUIPRegMappedAndELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    _log.info("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): START\n {0}\n", moduleOp);

    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    //
    // compute the size of each task list
    //

    mlir::Value dmaTasks;
    mlir::Value invariantTasks;
    mlir::Value variantTasks;
    mlir::Value actKernelInvocations;
    mlir::Value actKernelRanges;
    mlir::Value barrierTasks;

    for (auto op : funcOp.getOps<VPUIPRegMapped::NNDMAOp>()) {
        if (dmaCount == 0) {
            dmaTasks = op.getResult();
        }
        dmaCount++;
    }

    for (auto op : funcOp.getOps<VPUIPRegMapped::ConfigureBarrierOp>()) {
        if (barrierCount == 0) {
            barrierTasks = op.getResult();
        }
        barrierCount++;
    }

    for (auto op : funcOp.getOps<VPUIPRegMapped::ActKernelRangeOp>()) {
        if (rangeCount == 0) {
            actKernelRanges = op.getResult();
        }
        rangeCount++;
    }

    for (auto op : funcOp.getOps<VPUIPRegMapped::ActKernelInvocationOp>()) {
        if (invoCount == 0) {
            actKernelInvocations = op.getResult();
        }
        invoCount++;
    }

    auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

    // create MappedInferenceOp
    VPUIPRegMapped::MappedInferenceOp mappedInferenceOp =
            builderFunc.create<VPUIPRegMapped::MappedInferenceOp>(mlir::UnknownLoc::get(ctx), trivialIndexType,
                                                                  dmaTasks,        // mlir::Value dmaList
                                                                  invariantTasks,  // mlir::Value invariantList
                                                                  variantTasks,    // mlir::Value variantList
                                                                  actKernelRanges,
                                                                  actKernelInvocations,  // mlir::Value actInvocations
                                                                  barrierTasks,          // mlir::Value barrierList
                                                                  dmaCount,              // uint32_t dmaCount
                                                                  0,                     // uint32_t invariantCount
                                                                  0,                     // uint32_t variantCount
                                                                  rangeCount,            // uint32_t actKernelRanges
                                                                  invoCount,    // uint32_t actKernelInvocations
                                                                  barrierCount  // uint32_t barrierCount
            );

    //
    // Sections Creation
    //

    mlir::Value nndmaSectionOpValue = createSection<vpux::VPUIPRegMapped::NNDMAOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.dmaTasks", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC | vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);
    mlir::Value barrierSectionOpValue = createSection<vpux::VPUIPRegMapped::ConfigureBarrierOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.BarrierConfigs", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);
    mlir::Value kernelTextSectionOpValue =
            createSection<vpux::VPUIPRegMapped::DeclareKernelTextOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.KernelText", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    mlir::Value kernelDataSectionOpValue =
            createSection<vpux::VPUIPRegMapped::DeclareKernelArgsOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.KernelData", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    mlir::Value kernelParamsSectionOpValue = createSection<vpux::VPUIPRegMapped::KernelParamsOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.KernelParams", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    mlir::Value actKernelRangesSectionOpValue =
            createSection<vpux::VPUIPRegMapped::ActKernelRangeOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.ActKernelRanges", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    mlir::Value actKernelInvosSectionOpValue =
            createSection<vpux::VPUIPRegMapped::ActKernelInvocationOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.ActKernelInvocations", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    mlir::Value mappedInferenceSectionOpValue =
            createSection<vpux::VPUIPRegMapped::MappedInferenceOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.MappedInference", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    auto metadataSectionOp = builderFunc.create<ELF::CreateMetadataSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),       // mlir::Type
            ".metadata",                            // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,  // vpux::ELF::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,               // int64_t secInfo,
            64                                      // int64_t secAddrAlign
    );

    mlir::Block* blkMetadataSec = &(metadataSectionOp.aRegion().emplaceBlock());

    mlir::OpBuilder builderMetadataSec(blkMetadataSec, blkMetadataSec->end());

    builderMetadataSec.create<VPUIPRegMapped::NetworkMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType);

    _log.info("Convert2VPUIPRegMappedAndELFPass, after sections creation:\n {0} \n", moduleOp);

    //
    // Create Symbols for the relevant sections
    //

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    auto dmaSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),                // mlir::Type
                                              nndmaSectionOpValue,                            // mlir::Value inputArg
                                              isBuiltin,                                      // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_dmaSection0"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_dmaSection0"] = dmaSectionSym.getResult();

    auto barrierSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),                   // mlir::Type
                                              barrierSectionOpValue,                             // mlir::Value inputArg
                                              isBuiltin,                                         // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_barrierSection"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_barrierSection"] = barrierSectionSym.getResult();

    auto actKernelRangeSectionSym = builderFunc.create<ELF::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SymbolType::get(ctx),                          // mlir::Type
            actKernelRangesSectionOpValue,                            // mlir::Value inputArg
            isBuiltin,                                                // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_actKernelRangeSection"),  // mlir::StringAttr
            typeSym,                                                  // vpux::ELF::SymbolTypeAttrAttr
            sizeSym,                                                  // size
            valueSym                                                  // value
    );

    symbolMap["sym_actKernelRangeSection"] = actKernelRangeSectionSym.getResult();

    auto actKernelInvoSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),                  // mlir::Type
                                              actKernelInvosSectionOpValue,                     // mlir::Value inputArg
                                              isBuiltin,                                        // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_actKernelInvo"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_actKernelInvo"] = actKernelInvoSectionSym.getResult();

    auto kernelTextSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),  // mlir::Type
                                              kernelTextSectionOpValue,         // mlir::Value inputArg
                                              isBuiltin,                        // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_kernelTextSection"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_kernelTextSection"] = kernelTextSectionSym.getResult();

    auto kernelDataSectionSym =
            builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                              vpux::ELF::SymbolType::get(ctx),  // mlir::Type
                                              kernelDataSectionOpValue,         // mlir::Value inputArg
                                              isBuiltin,                        // mlir::UnitAttr
                                              mlir::StringAttr::get(ctx, "sym_kernelDataSection"),  // mlir::StringAttr
                                              typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                              sizeSym,  // size
                                              valueSym  // value
            );

    symbolMap["sym_kernelDataSection"] = kernelDataSectionSym.getResult();

    auto kernelParamsSectionSym = builderFunc.create<ELF::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELF::SymbolType::get(ctx),                        // mlir::Type
            kernelParamsSectionOpValue,                             // mlir::Value inputArg
            isBuiltin,                                              // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_kernelParamsSection"),  // mlir::StringAttr
            typeSym,                                                // vpux::ELF::SymbolTypeAttrAttr
            sizeSym,                                                // size
            valueSym                                                // value
    );

    symbolMap["sym_kernelParamsSection"] = kernelParamsSectionSym.getResult();

    //
    // Creation of SymTabs
    //

    createNetworkIOSymtab(funcOp, ctx, cnnOp);
    bufferSymTabValue = createBuffersSecAndSymtab(funcOp, ctx);
    CMXMappingSymtabValue = createCMXMappingSymtab(funcOp, ctx);

    auto tasksSymTabOp =
            builderFunc.create<ELF::CreateSymbolTableSectionOp>(mlir::UnknownLoc::get(ctx),
                                                                vpux::ELF::SectionType::get(ctx),  // mlir::Type
                                                                ".symtab.tasks",  // llvm::StringRef secName,
                                                                vpux::ELF::SectionFlagsAttr::SHF_NONE,
                                                                isBuiltin  // mlir::UnitAttr
            );

    tasksSymTabValue = tasksSymTabOp.getResult();

    mlir::Region& regTasksSymTabOp = tasksSymTabOp.getOperation()->getRegion(0);
    mlir::Block* blkTasksSymTabOp = new mlir::Block();
    regTasksSymTabOp.push_back(blkTasksSymTabOp);
    mlir::OpBuilder builderTasksSymTab(blkTasksSymTabOp, blkTasksSymTabOp->begin());

    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(), dmaSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(), barrierSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     kernelTextSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     kernelDataSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     kernelParamsSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     actKernelRangeSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     actKernelInvoSectionSym.getResult());

    auto vpux_entry_symbolTypeAttr = vpux::ELF::SymbolTypeAttrAttr::get(ctx, vpux::ELF::SymbolTypeAttr::VPU_STT_ENTRY);
    auto mappedInferenceSym = builderTasksSymTab.create<ELF::SymbolOp>(
            builderTasksSymTab.getUnknownLoc(),
            vpux::ELF::SymbolType::get(ctx),                      // mlir::Type
            mappedInferenceOp.getResult(),                        // mlir::Value inputArg
            isBuiltin,                                            // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "MappedInference_entry"),  // mlir::StringAttr
            vpux_entry_symbolTypeAttr,                            // vpux::ELF::SymbolTypeAttrAttr
            sizeSym,                                              // size
            valueSym                                              // value
    );

    symbolMap["MappedInference_entry"] = mappedInferenceSym.getResult();

    _log.info("Convert2VPUIPRegMappedAndELFPass, after symtabs creation:\n {0} \n", moduleOp);

    //
    // create general relocs for the tasks
    //

    createDMARelocs(funcOp, ctx, nndmaSectionOpValue);
    _log.info("Convert2VPUIPRegMappedAndELFPass, after DMA Relocs creation:\n {0} \n", moduleOp);

    createKernelParamsRelocs(funcOp, ctx, kernelParamsSectionOpValue);
    createActKernelRelocs(funcOp, ctx, actKernelRangesSectionOpValue, actKernelInvosSectionOpValue);
    setupActKernelRtConfigs(funcOp, ctx, mappedInferenceSectionOpValue);
    _log.info("Convert2VPUIPRegMappedAndELFPass, after ActKernel Relocs creation:\n {0} \n", moduleOp);

    //
    // create relocs for the tasks in MappedInference
    //

    ELF::CreateRelocationSectionOp createMappedInfRelocationSectionOp =
            builderFunc.create<ELF::CreateRelocationSectionOp>(
                    mlir::UnknownLoc::get(ctx),
                    vpux::ELF::SectionType::get(ctx),           // mlir::Type
                    ".rlt.MappedInference",                     // llvm::StringRef secName,
                    tasksSymTabValue,                           // sourceSymbolTableSection,
                    mappedInferenceSectionOpValue,              // targetSection,
                    vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK  // vpux::ELF::SectionFlagsAttr secFlags,
            );

    mlir::Region& regMappedInfRelocSec = createMappedInfRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkMappedInfRelocSec = new mlir::Block();
    regMappedInfRelocSec.push_back(blkMappedInfRelocSec);

    mlir::OpBuilder builderMappedInfRelocSec(blkMappedInfRelocSec, blkMappedInfRelocSec->begin());

    if (dmaCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(host_parsing::MappedInference, dmaTasks) +
                        offsetof(host_parsing::TaskReference<host_parsing::DmaWrapper>, address),
                vpux::ELF::RelocationTypeAttr::R_VPU_64, dmaSectionSym.getResult(), 0);
    }

    if (barrierCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(host_parsing::MappedInference, barrierConfigs) +
                        offsetof(host_parsing::TaskReference<host_parsing::BarrierWrapper>, address),
                vpux::ELF::RelocationTypeAttr::R_VPU_64, barrierSectionSym.getResult(), 0);
    }

    if (rangeCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(host_parsing::MappedInference, actKRanges) +
                        offsetof(host_parsing::TaskReference<host_parsing::ActKernelRangeWrapper>, address),
                vpux::ELF::RelocationTypeAttr::R_VPU_64, actKernelRangeSectionSym.getResult(), 0);
    }

    if (invoCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(host_parsing::MappedInference, actKInvocations) +
                        offsetof(host_parsing::TaskReference<host_parsing::ActKernelInvocationWrapper>, address),
                vpux::ELF::RelocationTypeAttr::R_VPU_64, actKernelInvoSectionSym.getResult(), 0);
    }

    _log.info("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): FINISH\n {0}\n", moduleOp);
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
