//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/nn_public.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <kernels/inc/common_types.h>

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

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
                              vpux::ELF::SectionTypeAttr secType, vpux::ELF::SectionFlagsAttr secFlags,
                              elf::Elf_Word secAlign = elf::VPU_SH_ADDR_ALIGN_FOR_VPU);

    mlir::Value createCMXMappingSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx);
    mlir::Value lookupELFSymbol(mlir::Value& symtabValue, mlir::Value& sym_input_value);
    mlir::Value createBuffersSecAndSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx);
    void createNetworkIOSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx, vpux::IE::CNNNetworkOp cnnOp);
    void createDMARelocs(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& dmaSectionValue);
    void createKernelParamsRelocs(mlir::FuncOp func);
    void createActKernelRelocs(mlir::FuncOp func);
    void setupActKernelRtConfigs(mlir::FuncOp func, mlir::ModuleOp moduleOp, mlir::MLIRContext* ctx);
    void createDPURelocs(mlir::FuncOp func);

    void safeRunOnModule() final;

private:
    Logger _log;

    vpux::ELF::RelocationManager relocationManager;

    mlir::Value networkInputSymTabValue, networkOutputSymTabValue;

    mlir::Value tasksSymTabValue, bufferSymTabValue, CMXMappingSymtabValue;

    mlir::Value mappedInferenceSectionOpValue;

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
    unsigned int variantCount = 0;
    unsigned int invariantCount = 0;
};

// createSection() creates an ELF::CreateSectionOp and puts into its body
//   an ELF.PutOpInSectionOp instruction for each object of type DerivedOpType
//   from func (a FuncOp).
template <typename DerivedOpType, typename CreateSectionOpType>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                            std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                            vpux::ELF::SectionFlagsAttr secFlags,
                                                            elf::Elf_Word secAlign) {
    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    size_t opAlignmentRequirements = DerivedOpType::getAlignmentRequirements();
    size_t secAlignReq = vpux::ELF::math::lcm(secAlign, opAlignmentRequirements);

    CreateSectionOpType elfCreateSectionOp =
            builderFunc.create<CreateSectionOpType>(mlir::UnknownLoc::get(ctx),
                                                    sectionType,               // mlir::Type
                                                    secNameStr,                // llvm::StringRef secName,
                                                    secType,                   // vpux::ELF::SectionTypeAttr secType,
                                                    secFlags,                  // vpux::ELF::SectionFlagsAttr secFlags,
                                                    elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                    secAlignReq                // int64_t secAddrAlign
            );

    auto builder = mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());

    size_t offsetTracker = secAlignReq;

    auto ops = func.getOps<DerivedOpType>();
    if (ops.empty()) {
        return elfCreateSectionOp.getResult();
    }

    for (DerivedOpType op : ops) {
        if (auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(&op)) {
            if (declareBufferOp->section() != vpux::VPURT::BufferSection::DDR) {
                continue;
            }
        }

        if (auto binaryOp = mlir::dyn_cast<vpux::ELF::BinaryOpInterface>(op.getOperation())) {
            size_t paddingRequired = offsetTracker % binaryOp.getAlignmentRequirements();
            if (paddingRequired) {
                auto off = secAlignReq - paddingRequired;
                builder.template create<ELF::PadOp>(builder.getUnknownLoc(), off, nullptr);
                offsetTracker += off;
            }

            builder.template create<ELF::PutOpInSectionOp>(builder.getUnknownLoc(), op.getResult());
            offsetTracker += binaryOp.getBinarySize();
        } else {
            VPUX_THROW("createSection: Op does not implement BinaryOp Interface");
        }
    }

    return elfCreateSectionOp.getResult();
}

template <>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection<Const::DeclareOp, ELF::CreateSectionOp>(
        mlir::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
        vpux::ELF::SectionFlagsAttr secFlags, elf::Elf_Word secAlign) {
    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    auto elfCreateSectionOp =
            builderFunc.create<ELF::CreateSectionOp>(mlir::UnknownLoc::get(ctx),
                                                     sectionType,               // mlir::Type
                                                     secNameStr,                // llvm::StringRef secName,
                                                     secType,                   // vpux::ELF::SectionTypeAttr secType,
                                                     secFlags,                  // vpux::ELF::SectionFlagsAttr secFlags,
                                                     elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                     secAlign                   // int64_t secAddrAlign
            );

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    mlir::Value constSecValue = elfCreateSectionOp.getResult();

    builderFunc.create<ELF::SymbolOp>(builderFunc.getUnknownLoc(),
                                      vpux::ELF::SymbolType::get(ctx),                 // mlir::Type
                                      constSecValue,                                   // mlir::Value inputArg
                                      isBuiltin,                                       // mlir::UnitAttr
                                      mlir::StringAttr::get(ctx, "sym_constSection"),  // mlir::StringAttr
                                      typeSym,                                         // vpux::ELF::SymbolTypeAttrAttr
                                      sizeSym,                                         // size
                                      valueSym                                         // value
    );

    auto builder = mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());

    for (Const::DeclareOp op : func.getOps<Const::DeclareOp>()) {
        builder.create<ELF::PutOpInSectionOp>(builder.getUnknownLoc(),  // endOp->getLoc(),
                                              op.getResult()            // mlir::Value inputArg
        );
    }

    return elfCreateSectionOp.getResult();
}

mlir::Value Convert2VPUIPRegMappedAndELFPass::createBuffersSecAndSymtab(mlir::FuncOp func, mlir::MLIRContext* ctx) {
    mlir::Value bufferSectionOpValue = createSection<vpux::VPURT::DeclareBufferOp, ELF::CreateLogicalSectionOp>(
            func, ctx, ".data.BuffersIO", vpux::ELF::SectionTypeAttr::SHT_NOBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC);

    auto constSecValue = createSection<vpux::Const::DeclareOp, ELF::CreateSectionOp>(
            func, ctx, ".data.ConstIO", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC);

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

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

    ELF::ElfSectionInterface constSecInterface =
            mlir::dyn_cast<ELF::ElfSectionInterface>(constSecValue.getDefiningOp());

    builderBufferSymTab.create<ELF::PutOpInSectionOp>(builderBufferSymTab.getUnknownLoc(),
                                                      ELF::RelocationManager::getSymbol(constSecInterface));

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

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    std::vector<mlir::Value> inputSyms;
    std::vector<mlir::Value> outputSyms;

    mlir::IntegerType uint64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    for (auto funcArg : func.getArguments()) {
        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        vpux::ELF::SymbolTypeAttrAttr typeSym;
        mlir::IntegerAttr valueSym;
        mlir::UnitAttr isBuiltin = nullptr;

        auto argNDType = funcArg.getType().cast<vpux::NDTypeInterface>();
        mlir::IntegerAttr sizeSym = mlir::IntegerAttr::get(uint64Type, argNDType.getTotalAllocSize().count());

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

void Convert2VPUIPRegMappedAndELFPass::createKernelParamsRelocs(mlir::FuncOp func) {
    auto kernelParamsOps = func.getOps<vpux::VPUIPRegMapped::KernelParamsOp>();

    if (kernelParamsOps.empty()) {
        return;
    }

    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTab;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;
    mlir::Value kernelParamsSectionSym = symbolMap["sym_kernelParamsSection"];

    // All the Kenel Params stuctures are serialized in a single section, in a continuous manner
    // All the relocations (excluding I/O ones), relocated addresses belonging to the same section as the target section
    targetSection = relocationManager.getSection((*kernelParamsOps.begin()).getResult());
    symTab = relocationManager.getSymTab((*kernelParamsOps.begin()));
    relocSection = relocationManager.getRelocSection(targetSection, symTab);

    auto paramsAutoRelocBuilder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    size_t total_addend = 0;

    for (auto kernelParamsOp : kernelParamsOps) {
        auto partial_addend = total_addend + kernelParamsOp.getParamsStructSize();

        auto kernelInputs = kernelParamsOp.inputs();

        // input addr
        for (auto kernelInput : kernelInputs) {
            symTab = relocationManager.getSymTab(kernelInput);

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            auto kernelInputBinaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(kernelInput.getDefiningOp());

            size_t addend = 0;

            if (kernelInputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELF::getOffsetOfOpInSection(kernelInput);
            } else {
                auto kernelInputSection = relocationManager.getSection(kernelInput);
                sourceSym = ELF::RelocationManager::getSymbol(kernelInputSection);
                mlir::Value kernelInputSectionValue = kernelInputSection.getOperation()->getResult(0);
                addend = ELF::getOffsetOfOpInSection(kernelInput, kernelInputSectionValue);
            }

            builder.create<ELF::RelocOp>(kernelInput.getLoc(), kernelParamsOp, kernelInput,
                                         vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        }

        // input Dims addr
        for (auto kernelInputsIt : kernelInputs | indexed) {
            paramsAutoRelocBuilder.create<ELF::RelocImmOffsetOp>(
                    kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                    kernelInputsIt.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dimsAddr),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);

            partial_addend += sizeof(int32_t) * getShape(kernelInputsIt.value()).size();
        }

        // input Strides addr
        for (auto kernelInputsIt : kernelInputs | indexed) {
            paramsAutoRelocBuilder.create<ELF::RelocImmOffsetOp>(
                    kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                    kernelInputsIt.index() * sizeof(sw_params::MemRefData) +
                            offsetof(sw_params::MemRefData, stridesAddr),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);

            partial_addend += sizeof(int64_t) * getMemStrides(kernelInputsIt.value()).size();
        }

        auto kernelOutputs = kernelParamsOp.outputs();
        auto kernelInputsSize = kernelInputs.size();

        // output addr
        for (auto kernelOutput : kernelOutputs) {
            symTab = relocationManager.getSymTab(kernelOutput);

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            auto kernelOutputBinaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(kernelOutput.getDefiningOp());

            size_t addend = 0;

            if (kernelOutputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELF::getOffsetOfOpInSection(kernelOutput);
            } else {
                auto kernelOutputSection = relocationManager.getSection(kernelOutput);
                sourceSym = ELF::RelocationManager::getSymbol(kernelOutputSection);
                mlir::Value kernelOutputSectionValue = kernelOutputSection.getOperation()->getResult(0);
                addend = ELF::getOffsetOfOpInSection(kernelOutput, kernelOutputSectionValue);
            }

            builder.create<ELF::RelocOp>(kernelOutput.getLoc(), kernelParamsOp, kernelOutput,
                                         vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        }

        // output Dims addr
        for (auto kernelOutputsIt : kernelOutputs | indexed) {
            paramsAutoRelocBuilder.create<ELF::RelocImmOffsetOp>(
                    kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                    (kernelInputsSize + kernelOutputsIt.index()) * sizeof(sw_params::MemRefData) +
                            offsetof(sw_params::MemRefData, dimsAddr),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);

            partial_addend += sizeof(int32_t) * getShape(kernelOutputsIt.value()).size();
        }

        // output Strides addr
        for (auto kernelOutputsIt : kernelOutputs | indexed) {
            paramsAutoRelocBuilder.create<ELF::RelocImmOffsetOp>(
                    kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                    (kernelInputsSize + kernelOutputsIt.index()) * sizeof(sw_params::MemRefData) +
                            offsetof(sw_params::MemRefData, stridesAddr),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32, kernelParamsSectionSym, partial_addend);
        }

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

void Convert2VPUIPRegMappedAndELFPass::createActKernelRelocs(mlir::FuncOp func) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTab;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;
    size_t total_addend = 0;

    // range relocs
    auto actKernelRangeOps = func.getOps<vpux::VPUIPRegMapped::ActKernelRangeOp>();
    for (auto actKernelRangeOp : actKernelRangeOps) {
        targetSection = relocationManager.getSection(actKernelRangeOp.getResult());

        auto kernelText = actKernelRangeOp.kernel_text_index();

        symTab = relocationManager.getSymTab(kernelText);

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelTextSection"].getDefiningOp());

        builder.create<ELF::RelocOp>(kernelText.getLoc(), actKernelRangeOp, kernelText,
                                     vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym, total_addend);

        auto declareKernelTextOp = llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareKernelTextOp>(
                actKernelRangeOp.kernel_text_index().getDefiningOp());
        total_addend += declareKernelTextOp.getBinarySize();
    }

    // invo relocs
    auto actKernelInvoOps = func.getOps<vpux::VPUIPRegMapped::ActKernelInvocationOp>();

    size_t dataSec_total_addend = 0;
    size_t paramsSec_total_addend = 0;

    for (auto actKernelInvoOp : actKernelInvoOps) {
        auto actKernelInvoOpIndex = actKernelInvoOp.index().getType().cast<vpux::VPUIPRegMapped::IndexType>();
        auto associatedRangeOp =
                llvm::dyn_cast<vpux::VPUIPRegMapped::ActKernelRangeOp>(actKernelInvoOp.range_index().getDefiningOp());

        targetSection = relocationManager.getSection(actKernelInvoOp.getResult());

        // range reloc
        symTab = relocationManager.getCMXSymTab();

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        builder.create<ELF::RelocOp>(
                relocSection.getLoc(), actKernelInvoOp.getResult(), associatedRangeOp.getResult(),
                vpux::ELF::RelocationTypeAttr::R_VPU_32_RTM,
                elfCMXMappingSyms[static_cast<int>(vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_ACT)].getResult(),
                sizeof(nn_public::ActKernelRange));

        // data section reloc
        auto kernelArgs = associatedRangeOp.kernel_args_index();

        symTab = relocationManager.getSymTab(kernelArgs);

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelDataSection"].getDefiningOp());

        builder.create<ELF::RelocImmOffsetOp>(kernelArgs.getLoc(), actKernelInvoOp.getResult(),
                                              offsetof(nn_public::ActKernelInvocation, data_window_base),
                                              vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym, dataSec_total_addend);

        auto declareKernelArgsOp = llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareKernelArgsOp>(
                associatedRangeOp.kernel_args_index().getDefiningOp());
        dataSec_total_addend += declareKernelArgsOp.getBinarySize();

        // params reloc
        vpux::VPUIPRegMapped::KernelParamsOp associatedKernelParamsOp;
        auto kernelParamsOps = func.getOps<vpux::VPUIPRegMapped::KernelParamsOp>();
        for (auto kernelParamsOp : kernelParamsOps) {
            auto kernelParamsOpIndex = kernelParamsOp.index().getType().cast<vpux::VPUIPRegMapped::IndexType>();
            if (kernelParamsOpIndex.getValue() == actKernelInvoOpIndex.getValue()) {
                associatedKernelParamsOp = kernelParamsOp;
                break;
            }
        }

        symTab = relocationManager.getSymTab(associatedKernelParamsOp.getResult());

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelParamsSection"].getDefiningOp());

        builder.create<ELF::RelocImmOffsetOp>(associatedKernelParamsOp.getLoc(), actKernelInvoOp.getResult(),
                                              offsetof(nn_public::ActKernelInvocation, kernel_args),
                                              vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym,
                                              paramsSec_total_addend);

        paramsSec_total_addend += associatedKernelParamsOp.getBinarySize();
    }
}

void Convert2VPUIPRegMappedAndELFPass::setupActKernelRtConfigs(mlir::FuncOp func, mlir::ModuleOp moduleOp,
                                                               mlir::MLIRContext* ctx) {
    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    auto mappedInferenceOps = func.getOps<VPUIPRegMapped::MappedInferenceOp>();

    VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");

    auto mappedInferenceOp = *(mappedInferenceOps.begin());

    if (mappedInferenceOp.actKernelInvocationsCount() == 0) {
        return;
    }
    auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");

    auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::FuncOp>("runtime");

    mlir::Value nnActEntryText;
    ELF::ElfSectionInterface actKRtConfigSec;

    if (runtimeKernelFunction) {
        const auto kernelElf =
                std::string(runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code").getValue());

        auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

        auto nnActEntryTextOp = builderFunc.create<VPUIPRegMapped::DeclareKernelTextOp>(
                builderFunc.getUnknownLoc(), trivialIndexType, mlir::StringAttr::get(ctx, kernelElf));

        nnActEntryText = nnActEntryTextOp.getResult();

        actKRtConfigSec = builderFunc.create<ELF::CreateSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),          // mlir::Type
                ".text.actKernelRtConfigSec",              // llvm::StringRef secName,
                vpux::ELF::SectionTypeAttr::SHT_PROGBITS,  // vpux::ELF::SectionTypeAttr secType,
                vpux::ELF::SectionFlagsAttr::SHF_NONE,     // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                  // int64_t secInfo,
                1024                                       // int64_t secAddrAlign
        );

    } else {
        const auto bufferMemrefShape = SmallVector<int64_t>{262144};
        auto DDRNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
        const auto DDRSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, DDRNameAttr);

        unsigned int perm[1] = {0};

        auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), ctx);

        auto memrefType = mlir::MemRefType::get(bufferMemrefShape, mlir::IntegerType::get(ctx, 32), map, DDRSymbolAttr);

        auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                          memrefType,                 // Type
                                                                          VPURT::BufferSection::DDR,  // Buffer Type
                                                                          0                           // byteOffset
        );

        nnActEntryText = declareBufferOp.getResult();

        actKRtConfigSec = builderFunc.create<ELF::CreateLogicalSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),        // mlir::Type
                ".bss.actKernelRtConfigSec",             // llvm::StringRef secName,
                vpux::ELF::SectionTypeAttr::SHT_NOBITS,  // vpux::ELF::SectionTypeAttr secType,
                vpux::ELF::SectionFlagsAttr::SHF_NONE,   // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                // int64_t secInfo,
                1024                                     // int64_t secAddrAlign
        );
    }

    // Depending on the case, the Section must be binary or logical
    // Refactor such that it comprises both logic

    auto builderElfSectionOpReg = mlir::OpBuilder::atBlockEnd(actKRtConfigSec.getBlock());

    builderElfSectionOpReg.create<ELF::PutOpInSectionOp>(builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                                                         nnActEntryText  // mlir::Value inputArg
    );

    mlir::Value actKRtConfigSecValue = actKRtConfigSec.getOperation()->getResult(0);

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

    auto builderActKRtConfigSymTab = mlir::OpBuilder::atBlockEnd(actKRtConfigSymTab.getBlock());

    builderActKRtConfigSymTab.create<ELF::PutOpInSectionOp>(builderActKRtConfigSymTab.getUnknownLoc(),
                                                            actKRtConfigSym.getResult());

    mlir::Value actKRtConfigSymValue = actKRtConfigSym.getResult();

    ELF::ElfSectionInterface mappedInferenceSec =
            mlir::cast<ELF::ElfSectionInterface>(mappedInferenceSectionOpValue.getDefiningOp());

    VPUX_THROW_UNLESS(mappedInferenceSec != nullptr, "CreateActKernelConfig: MappedInference section is null");

    auto relocSection = relocationManager.getRelocSection(mappedInferenceSec, actKRtConfigSymTab);

    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    for (auto mappedInferenceOp : mappedInferenceOps) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(nn_public::MappedInference, shv_rt_configs) +
                        offsetof(nn_public::NNShaveRuntimeConfigs, act_rt_window_base),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, actKRtConfigSymValue, 0);
    }
}

void Convert2VPUIPRegMappedAndELFPass::createDPURelocs(mlir::FuncOp func) {
    auto invariants = func.getOps<VPUIPRegMapped::DPUInvariantOp>();

    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTabOfInput;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;
    VPURT::DeclareBufferOp declarator;

    ELF::SymbolOp weightTableStartSym;
    uint64_t weightTableStartAddend;

    // TODO: E#54007 currently ignoring sparsity and SOH/SOK.
    for (auto invariant : invariants) {
        auto opType = invariant.task_type();

        auto result = invariant.index();
        targetSection = relocationManager.getSection(result);

        auto input = invariant.input();
        declarator = mlir::cast<VPURT::DeclareBufferOp>(input.getDefiningOp());

        symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                : relocationManager.getSymTab(input);

        relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = declarator.section() == VPURT::BufferSection::CMX_NN
                            ? elfCMXMappingSyms[static_cast<int>(
                                      vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                            : relocationManager.getSymbol(targetSection);

        auto regsOffset = offsetof(nn_public::DPUInvariant, registers_);

        // input relocs, relocating act_offset[0] registers
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::DPUInvariantRegisters, act_offset[0]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, declarator.byteOffset());
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::DPUInvariantRegisters, act_offset[1]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, declarator.byteOffset());
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::DPUInvariantRegisters, act_offset[2]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, declarator.byteOffset());
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::DPUInvariantRegisters, act_offset[3]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, declarator.byteOffset());

        // weights

        if (auto weights = invariant.weights()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            sourceSym = declarator.section() == VPURT::BufferSection::CMX_NN
                                ? elfCMXMappingSyms[static_cast<int>(
                                          vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                : relocationManager.getSymbol(targetSection);

            if (opType != VPUIP::NCETaskType::ELTWISE) {
                builder.create<ELF::RelocImmOffsetOp>(
                        weights.getLoc(), invariant, regsOffset + offsetof(nn_public::DPUInvariantRegisters, wt_offset),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, 0);
            } else {
                auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp()).byteOffset();
                auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.input().getDefiningOp()).byteOffset();

                // correlated with serializer, where rest of the offsets are expected to be directly filled, in
                // accordance with this if-then-else
                builder.create<ELF::RelocImmOffsetOp>(
                        invariant.input().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::DPUInvariantRegisters, act_offset[0]),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, std::min(actOffs, weightsOffs));
            }
        }

        auto output = invariant.output_buff();

        declarator = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

        symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                : relocationManager.getSymTab(input);

        relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

        builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = declarator.section() == VPURT::BufferSection::CMX_NN
                            ? elfCMXMappingSyms[static_cast<int>(
                                      vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                            : relocationManager.getSymbol(targetSection);

        // TODO:E#54007 sutpport multiclustering eventually. Find cleaner solution for this offsets magic
        int nCluster = 1;
        static constexpr uint32_t baseOffsets[4] = {offsetof(nn_public::DPUInvariantRegisters, base_adr[0]),
                                                    offsetof(nn_public::DPUInvariantRegisters, base_adr[1]),
                                                    offsetof(nn_public::DPUInvariantRegisters, base_adr[2]),
                                                    offsetof(nn_public::DPUInvariantRegisters, base_adr[3])};
        static constexpr uint32_t oduCastOffsets[4] = {offsetof(nn_public::DPUInvariantRegisters, odu_cast[0]),
                                                       offsetof(nn_public::DPUInvariantRegisters, odu_cast[1]),
                                                       offsetof(nn_public::DPUInvariantRegisters, odu_cast[2]),
                                                       offsetof(nn_public::DPUInvariantRegisters, odu_cast[3])};

        builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + baseOffsets[0],
                                              ELF::RelocationTypeAttr::R_VPU_32_MULTICAST_BASE, sourceSym,
                                              declarator.byteOffset());

        for (int id = 0; id < nCluster; ++id) {
            builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + baseOffsets[id + 1],
                                                  ELF::RelocationTypeAttr::R_VPU_32_MULTICAST_BASE, sourceSym,
                                                  declarator.byteOffset());

            builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + oduCastOffsets[id],
                                                  ELF::RelocationTypeAttr::R_VPU_DISP28_MULTICAST_OFFSET, sourceSym,
                                                  declarator.byteOffset());
            builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + oduCastOffsets[id],
                                                  ELF::RelocationTypeAttr::R_VPU_DISP4_MULTICAST_OFFSET_CMP, sourceSym,
                                                  declarator.byteOffset());
        }
        // wtable

        if (auto weightTable = invariant.weight_table()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weightTable.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            sourceSym = declarator.section() == VPURT::BufferSection::CMX_NN
                                ? elfCMXMappingSyms[static_cast<int>(
                                          vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)]
                                : relocationManager.getSymbol(targetSection);

            weightTableStartSym = sourceSym;
            weightTableStartAddend = declarator.byteOffset();
            builder.create<ELF::RelocImmOffsetOp>(weightTable.getLoc(), invariant,
                                                  regsOffset + offsetof(nn_public::DPUInvariantRegisters, weight_start),
                                                  ELF::RelocationTypeAttr::R_VPU_32, sourceSym,
                                                  declarator.byteOffset());
        }

        // variant to invariant relocation
        auto children = invariant.getResult().getUsers();
        for (auto child : children) {
            auto variant = mlir::dyn_cast<VPUIPRegMapped::DPUVariantOp>(child);
            if (variant == nullptr) {
                continue;
            }

            auto invariantVal = variant.Invariant();

            auto targetSection = relocationManager.getSection(variant.getResult());

            auto relocSection = relocationManager.getRelocSection(
                    targetSection, mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp()));

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            sourceSym = elfCMXMappingSyms[static_cast<int>(ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_IVAR)];

            builder.create<ELF::RelocOp>(invariant.getLoc(), variant, invariantVal,
                                         ELF::RelocationTypeAttr::R_VPU_32_RTM, sourceSym,
                                         sizeof(nn_public::DPUInvariant));

            if (invariant.weight_table()) {
                builder.create<ELF::RelocImmOffsetOp>(
                        variant.getLoc(), variant, offsetof(nn_public::DPUVariant, weight_table_offset_),
                        ELF::RelocationTypeAttr::R_VPU_32_SUM, weightTableStartSym, weightTableStartAddend);
            }
        }
    }

    return;
}

void Convert2VPUIPRegMappedAndELFPass::createDMARelocs(mlir::FuncOp funcOp, mlir::MLIRContext* ctx,
                                                       mlir::Value& dmaSectionValue) {
    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTab;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;

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

    auto dmaOps = funcOp.getOps<vpux::VPUIPRegMapped::NNDMAOp>();

    targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(dmaSectionValue.getDefiningOp());

    for (auto dmaOp : dmaOps) {
        // input addr
        if (auto dmaInputArg = dmaOp.input().dyn_cast<mlir::BlockArgument>()) {
            if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaInputArg)) {
                builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netInputSymValue,                         // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            } else if (mlir::Value netInputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaInputArg)) {
                builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, src),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netInputSymValue,                         // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            }
        } else {
            auto dmaInputArg_ = dmaOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
            if (dmaInputArg_ && (dmaInputArg_.getMemorySpace() == VPURT::BufferSection::NetworkInput)) {
                auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaInputArg_.sectionIndex().getValue());
                VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                  "Encountered DMA op {} with input {} which has multiple section indexes {}", dmaOp,
                                  dmaInputArg_, funcArgIndex);
                auto funcArg = funcOp.getArgument(funcArgIndex[0]);
                if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, funcArg)) {
                    builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, src),
                            vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                            netInputSymValue,                         // ::mlir::Value sourceSymbol
                            0                                         // int64_t addend
                    );
                }
            } else {
                auto dmaInput = dmaOp.input();

                symTab = relocationManager.getSymTab(dmaInput);

                relocSection = relocationManager.getRelocSection(targetSection, symTab);

                auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                auto dmaInputBinaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(dmaInput.getDefiningOp());

                size_t addend = 0;

                if (dmaInputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                    sourceSym = elfCMXMappingSyms[static_cast<int>(
                            vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                    addend = ELF::getOffsetOfOpInSection(dmaInput);
                } else {
                    auto dmaInputSection = relocationManager.getSection(dmaInput);
                    sourceSym = ELF::RelocationManager::getSymbol(dmaInputSection);
                    mlir::Value dmaInputSectionValue = dmaInputSection.getOperation()->getResult(0);
                    addend = ELF::getOffsetOfOpInSection(dmaInput, dmaInputSectionValue);
                }

                builder.create<ELF::RelocOp>(dmaInput.getLoc(), dmaOp, dmaInput,
                                             vpux::ELF::RelocationTypeAttr::R_VPU_64, sourceSym, addend);
            }
        }

        // output addr
        if (auto dmaOutputArg = dmaOp.output_buff().dyn_cast<mlir::BlockArgument>()) {
            if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaOutputArg)) {
                builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netOutputSymValue,                        // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            } else if (mlir::Value netOutputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaOutputArg)) {
                builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                        builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                        offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, dst),
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                        netOutputSymValue,                        // ::mlir::Value sourceSymbol
                        0                                         // int64_t addend
                );
            }
        } else {
            auto dmaOutputArg_ = dmaOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
            VPUX_THROW_UNLESS(dmaOutputArg_,
                              "Encountered DMA op {} with output {} which is neither mlir::BlockArgument, nor "
                              "VPURT::DeclareBufferOp",
                              dmaOp, dmaOutputArg_);

            if (dmaOutputArg_.getMemorySpace() == VPURT::BufferSection::NetworkOutput) {
                auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaOutputArg_.sectionIndex().getValue());
                VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                  "Encountered DMA op {} with output {} which has multiple secion indexes {}", dmaOp,
                                  dmaOutputArg_, funcArgIndex);
                auto funcArg = funcOp.getArgument(funcArgIndex[0] + funcOp.getNumArguments() - funcOp.getNumResults());
                if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, funcArg)) {
                    builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::DMATask, transaction_) + offsetof(nn_public::vpu_dma_descriptor_t, dst),
                            vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                            netOutputSymValue,                        // ::mlir::Value sourceSymbol
                            0                                         // int64_t addend
                    );
                }
            } else {
                auto dmaOutput = dmaOp.output_buff();

                symTab = relocationManager.getSymTab(dmaOutput);

                relocSection = relocationManager.getRelocSection(targetSection, symTab);

                auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                auto dmaOutputBinaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(dmaOutput.getDefiningOp());

                size_t addend = 0;

                if (dmaOutputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                    sourceSym = elfCMXMappingSyms[static_cast<int>(
                            vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                    addend = ELF::getOffsetOfOpInSection(dmaOutput);
                } else {
                    auto dmaOutputSection = relocationManager.getSection(dmaOutput);
                    sourceSym = ELF::RelocationManager::getSymbol(dmaOutputSection);
                    mlir::Value dmaOutputSectionValue = dmaOutputSection.getOperation()->getResult(0);
                    addend = ELF::getOffsetOfOpInSection(dmaOutput, dmaOutputSectionValue);
                }

                builder.create<ELF::RelocOp>(dmaOutput.getLoc(), dmaOp, dmaOutput,
                                             vpux::ELF::RelocationTypeAttr::R_VPU_64, sourceSym, addend);
            }
        }

        // link_address
        if (dmaCount > dmaOp.getType().getValue() + 1) {
            symTab = relocationManager.getCMXSymTab();

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            builder.create<ELF::RelocImmOffsetOp>(
                    relocSection.getLoc(), dmaOp, offsetof(nn_public::DMATask, transaction_),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32_RTM,
                    elfCMXMappingSyms[static_cast<int>(vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_DMA0)]
                            .getResult(),
                    sizeof(nn_public::DMATask));
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

    relocationManager.init(funcOp);

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

    for (auto op : funcOp.getOps<VPUIPRegMapped::DPUInvariantOp>()) {
        if (invariantCount == 0) {
            invariantTasks = op.getResult();
        }
        invariantCount++;
    }

    for (auto op : funcOp.getOps<VPUIPRegMapped::DPUVariantOp>()) {
        if (variantCount == 0) {
            variantTasks = op.getResult();
        }
        variantCount++;
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
                                                                  invariantCount,        // uint32_t invariantCount
                                                                  variantCount,          // uint32_t variantCount
                                                                  rangeCount,            // uint32_t actKernelRanges
                                                                  invoCount,    // uint32_t actKernelInvocations
                                                                  barrierCount  // uint32_t barrierCount
            );

    //
    // Sections Creation
    //

    mlir::Value nndmaSectionOpValue = createSection<vpux::VPUIPRegMapped::NNDMAOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.dmaTasks", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC);

    mlir::Value barrierSectionOpValue = createSection<vpux::VPUIPRegMapped::ConfigureBarrierOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.BarrierConfigs", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelTextSectionOpValue =
            createSection<vpux::VPUIPRegMapped::DeclareKernelTextOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.KernelText", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelDataSectionOpValue =
            createSection<vpux::VPUIPRegMapped::DeclareKernelArgsOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.KernelData", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelParamsSectionOpValue = createSection<vpux::VPUIPRegMapped::KernelParamsOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.KernelParams", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value actKernelRangesSectionOpValue =
            createSection<vpux::VPUIPRegMapped::ActKernelRangeOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.ActKernelRanges", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value actKernelInvosSectionOpValue =
            createSection<vpux::VPUIPRegMapped::ActKernelInvocationOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.ActKernelInvocations", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mappedInferenceSectionOpValue = createSection<vpux::VPUIPRegMapped::MappedInferenceOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.MappedInference", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value invariantsSection = createSection<vpux::VPUIPRegMapped::DPUInvariantOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.DPUInvariants", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value variantsSection = createSection<vpux::VPUIPRegMapped::DPUVariantOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.DPUVariants", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    auto metadataSectionOp = builderFunc.create<ELF::CreateMetadataSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),       // mlir::Type
            ".metadata",                            // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,  // vpux::ELF::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,               // int64_t secInfo,
            vpux::VPUIPRegMapped::NetworkMetadataOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto builderMetadataSec = mlir::OpBuilder::atBlockEnd(metadataSectionOp.getBlock());

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

    auto inVariantsSectionSym = builderFunc.create<ELF::SymbolOp>(
            mlir::UnknownLoc::get(ctx), vpux::ELF::SymbolType::get(ctx), invariantsSection, isBuiltin,
            mlir::StringAttr::get(ctx, "sym_inVariantsSection"), typeSym, sizeSym, valueSym);

    symbolMap["sym_inVariantsSection"] = inVariantsSectionSym.getResult();

    auto variantsSectionSym = builderFunc.create<ELF::SymbolOp>(
            mlir::UnknownLoc::get(ctx), vpux::ELF::SymbolType::get(ctx), variantsSection, isBuiltin,
            mlir::StringAttr::get(ctx, "sym_variantsSection"), typeSym, sizeSym, valueSym);

    symbolMap["sym_variantsSection"] = variantsSectionSym.getResult();
    //
    // Creation of SymTabs
    //

    createNetworkIOSymtab(funcOp, ctx, cnnOp);
    bufferSymTabValue = createBuffersSecAndSymtab(funcOp, ctx);
    CMXMappingSymtabValue = createCMXMappingSymtab(funcOp, ctx);

    ELF::CreateSymbolTableSectionOp CMXMappingSymtabOp =
            mlir::dyn_cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
    relocationManager.initCMXSymTab(CMXMappingSymtabOp);

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
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     inVariantsSectionSym.getResult());
    builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                     variantsSectionSym.getResult());

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

    createKernelParamsRelocs(funcOp);
    createActKernelRelocs(funcOp);
    setupActKernelRtConfigs(funcOp, moduleOp, ctx);
    _log.info("Convert2VPUIPRegMappedAndELFPass, after Shave Relocs creation:\n {0} \n", moduleOp);

    createDPURelocs(funcOp);
    _log.info("Convert2VPUIPRegMappedAndELFPass, after ActKernel Relocs creation:\n {0} \n", moduleOp);

    //
    // create relocs for the tasks in MappedInference
    //

    ELF::ElfSectionInterface targetSection =
            mlir::dyn_cast<ELF::ElfSectionInterface>(mappedInferenceSectionOpValue.getDefiningOp());
    ELF::CreateSymbolTableSectionOp symTab = tasksSymTabOp;
    ELF::CreateRelocationSectionOp relocSection = relocationManager.getRelocSection(targetSection, symTab);

    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    if (dmaCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), dmaTasks,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, dmaSectionSym.getResult(), 0);
    }

    if (barrierCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), barrierTasks,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, barrierSectionSym.getResult(), 0);
    }

    if (rangeCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), actKernelRanges,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, actKernelRangeSectionSym.getResult(), 0);
    }

    if (invoCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), actKernelInvocations,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, actKernelInvoSectionSym.getResult(), 0);
    }

    if (invariantCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), invariantTasks,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, inVariantsSectionSym.getResult(), 0);
    }

    if (variantCount > 0) {
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), variantTasks,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, variantsSectionSym.getResult(), 0);
    }

    _log.info("Convert2VPUIPRegMappsedAndELFPass::safeRunOnFunc(): FINISH\n {0}\n", moduleOp);
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
