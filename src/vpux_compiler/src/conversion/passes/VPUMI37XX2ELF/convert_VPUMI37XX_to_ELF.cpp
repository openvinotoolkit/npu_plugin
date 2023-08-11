//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/act_kernels/nce2p7.h"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU37XX/api/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <kernels/inc/common_types.h>

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/Hashing.h>

#include <limits>

using namespace vpux;

constexpr auto NNCMX_SLICE_SIZE = mvds::nce2p7::CMX_SLICE_SIZE;
constexpr auto ACT_SHAVE_STACK_SIZE = (8_KB).to<vpux::Byte>().count();
constexpr auto ACT_RT_CODE_BUFFER_SIZE = (1_MB).to<vpux::Byte>().count();

namespace {

//
// ConvertVPUMI37XX2ELFPass
//

class ConvertVPUMI37XX2ELFPass final : public ConvertVPUMI37XX2ELFBase<ConvertVPUMI37XX2ELFPass> {
public:
    explicit ConvertVPUMI37XX2ELFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename DerivedOpType, typename CreateSectionOpType>
    mlir::Value createSection(mlir::func::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr,
                              vpux::ELF::SectionTypeAttr secType, vpux::ELF::SectionFlagsAttr secFlags,
                              elf::Elf_Word secAlign = elf::VPU_SH_ADDR_ALIGN_FOR_VPU);
    mlir::SmallVector<mlir::Value> createDMASections(mlir::func::FuncOp& func, mlir::MLIRContext* ctx,
                                                     mlir::SmallVector<int64_t>& dmaCount,
                                                     mlir::Operation::operand_range dmaTasks);

    mlir::Value createCMXMappingSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx);
    mlir::Value lookupELFSymbol(mlir::Value& symtabValue, mlir::Value& sym_input_value);
    mlir::Value createBuffersSecAndSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx);
    void createNetworkIOSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx, vpux::IE::CNNNetworkOp cnnOp);
    void createDMARelocs(mlir::func::FuncOp& func, mlir::MLIRContext* ctx, mlir::SmallVector<int64_t>& dmaCount,
                         mlir::Operation::operand_range dmaTasks, mlir::SmallVector<mlir::Value>& dmaSectionValues);
    void createKernelParamsRelocs(mlir::func::FuncOp func);
    void createActKernelRelocs(mlir::func::FuncOp func);
    void setupActKernelRtConfigs(mlir::func::FuncOp func, mlir::ModuleOp moduleOp, mlir::MLIRContext* ctx);
    void createDPURelocs(mlir::func::FuncOp func);

    void safeRunOnModule() final;

    mlir::Value getNextDMATask(mlir::Value& currentTask) {
        for (auto taskUser : currentTask.getUsers()) {
            if (auto nextDMAOp = mlir::dyn_cast_or_null<VPUMI37XX::NNDMAOp>(taskUser)) {
                auto currentDMAOp = mlir::cast<VPUMI37XX::NNDMAOp>(currentTask.getDefiningOp());
                VPUX_THROW_UNLESS(nextDMAOp.port() == currentDMAOp.port(),
                                  "Port mismatch betwen tasks from same DMA list");
                VPUX_THROW_UNLESS(nextDMAOp.previousDMAIdx() == currentTask, "DMA list fatal link error");
                return nextDMAOp.getResult();
            }
        }
        return mlir::Value();
    }

private:
    vpux::ELF::RelocationManager relocationManager;

    mlir::Value networkInputSymTabValue, networkOutputSymTabValue, profOutputSymTabValue;

    mlir::Value tasksSymTabValue, bufferSymTabValue, CMXMappingSymtabValue;

    mlir::Value mappedInferenceSectionOpValue;

    std::map<std::string, mlir::Value> symbolMap;

    // map that correlates between Const::DeclareOp values and their ELF::SymbolOp value
    llvm::MapVector<mlir::Value, mlir::Value> constSymMap;

    // map that correlates between Const::DeclareOp values and their offset in the .data.const section
    llvm::MapVector<mlir::Value, size_t> constOffsetMap;

    std::vector<ELF::SymbolOp> elfCMXMappingSyms;
};

// createSection() creates an ELF::CreateSectionOp and puts into its body
//   an ELF.PutOpInSectionOp instruction for each object of type DerivedOpType
//   from func (a FuncOp).
template <typename DerivedOpType, typename CreateSectionOpType>
mlir::Value ConvertVPUMI37XX2ELFPass::createSection(mlir::func::FuncOp func, mlir::MLIRContext* ctx,
                                                    std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                    vpux::ELF::SectionFlagsAttr secFlags, elf::Elf_Word secAlign) {
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

        auto binaryOp = mlir::cast<vpux::ELF::BinaryOpInterface>(op.getOperation());
        size_t paddingRequired = offsetTracker % binaryOp.getAlignmentRequirements();
        if (paddingRequired) {
            auto off = secAlignReq - paddingRequired;
            builder.template create<ELF::PadOp>(builder.getUnknownLoc(), off, nullptr);
            offsetTracker += off;
        }

        builder.template create<ELF::PutOpInSectionOp>(builder.getUnknownLoc(), op.getResult());
        offsetTracker += binaryOp.getBinarySize();
    }

    return elfCreateSectionOp.getResult();
}

mlir::SmallVector<mlir::Value> ConvertVPUMI37XX2ELFPass::createDMASections(mlir::func::FuncOp& func,
                                                                           mlir::MLIRContext* ctx,
                                                                           mlir::SmallVector<int64_t>& dmaCount,
                                                                           mlir::Operation::operand_range dmaTasks) {
    mlir::SmallVector<mlir::Value> returnValues;

    if (dmaTasks.empty()) {
        return returnValues;
    }

    std::string secNameBaseStr = ".text.dmaTasks";
    vpux::ELF::SectionTypeAttr secType = vpux::ELF::SectionTypeAttr::SHT_PROGBITS;
    vpux::ELF::SectionFlagsAttr secFlags = vpux::ELF::SectionFlagsAttr::SHF_ALLOC;

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);
    size_t opAlignmentRequirements = VPUMI37XX::NNDMAOp::getAlignmentRequirements();
    size_t secAlignReq = vpux::ELF::math::lcm(elf::VPU_SH_ADDR_ALIGN_FOR_VPU, opAlignmentRequirements);

    // Firstly, create sections for all DMA ports to keep reloc logic simple
    // Empty sections will be removed by cleanup pass
    for (size_t listIdx = 0; listIdx < dmaCount.size(); ++listIdx) {
        auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());
        ELF::CreateSectionOp elfCreateSectionOp = builderFunc.create<ELF::CreateSectionOp>(
                mlir::UnknownLoc::get(ctx),
                sectionType,                               // mlir::Type
                secNameBaseStr + std::to_string(listIdx),  // llvm::StringRef secName,
                secType,                                   // vpux::ELF::SectionTypeAttr secType,
                secFlags,                                  // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                  // int64_t secInfo,
                secAlignReq                                // int64_t secAddrAlign
        );

        mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());
        returnValues.push_back(elfCreateSectionOp.getResult());
    }

    // Secondly, populate sections with the corresponding DMA tasks
    for (auto listHead : dmaTasks) {
        size_t offsetTracker = secAlignReq;
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).port();
        auto listElemCount = dmaCount[listIdx];
        auto builder = mlir::OpBuilder::atBlockEnd(
                mlir::cast<ELF::CreateSectionOp>(returnValues[listIdx].getDefiningOp()).getBlock());

        for (auto dmaTaskIdx = 0; dmaTaskIdx < listElemCount; ++dmaTaskIdx) {
            auto nndmaOp = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp());
            auto binaryOp = mlir::cast<vpux::ELF::BinaryOpInterface>(nndmaOp.getOperation());
            size_t paddingRequired = offsetTracker % binaryOp.getAlignmentRequirements();
            if (paddingRequired) {
                auto off = secAlignReq - paddingRequired;
                builder.template create<ELF::PadOp>(builder.getUnknownLoc(), off, nullptr);
                offsetTracker += off;
            }

            builder.template create<ELF::PutOpInSectionOp>(builder.getUnknownLoc(), nndmaOp.getResult());
            offsetTracker += binaryOp.getBinarySize();

            listHead = getNextDMATask(listHead);
        }
    }

    return returnValues;
}  // namespace

template <>
mlir::Value ConvertVPUMI37XX2ELFPass::createSection<Const::DeclareOp, ELF::CreateSectionOp>(
        mlir::func::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
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

mlir::Value ConvertVPUMI37XX2ELFPass::createBuffersSecAndSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx) {
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

mlir::Value ConvertVPUMI37XX2ELFPass::createCMXMappingSymtab(mlir::func::FuncOp funcOp, mlir::MLIRContext* ctx) {
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

void ConvertVPUMI37XX2ELFPass::createNetworkIOSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx,
                                                     vpux::IE::CNNNetworkOp cnnOp) {
    SmallVector<vpux::IE::DataInfoOp, 1> dataInfoOpInVec = cnnOp.getInputsInfo();
    SmallVector<vpux::IE::DataInfoOp, 1> dataInfoOpOutVec = cnnOp.getOutputsInfo();
    SmallVector<vpux::IE::DataInfoOp, 1> dataInfoOpProfilingOutVec = cnnOp.getProfilingOutputsInfo();

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    std::vector<mlir::Value> inputSyms;
    std::vector<mlir::Value> outputSyms;
    std::vector<mlir::Value> profOutputSyms;

    mlir::IntegerType uint64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    for (auto funcArg : func.getArguments()) {
        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        vpux::ELF::SymbolTypeAttrAttr typeSym;
        mlir::IntegerAttr valueSym;
        mlir::UnitAttr isBuiltin = nullptr;

        auto argNDType = funcArg.getType().cast<vpux::NDTypeInterface>();
        mlir::IntegerAttr sizeSym = mlir::IntegerAttr::get(uint64Type, argNDType.getTotalAllocSize().count());

        mlir::StringAttr nameSym;
        std::vector<mlir::Value>* symsVecPtr = nullptr;
        auto index = funcArg.getArgNumber();
        if (index < dataInfoOpInVec.size()) {
            symsVecPtr = &inputSyms;
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpInVec[index].name());
        } else if (index < (dataInfoOpInVec.size() + dataInfoOpOutVec.size())) {
            symsVecPtr = &outputSyms;
            index -= dataInfoOpInVec.size();
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpOutVec[index].name());
        } else {
            symsVecPtr = &profOutputSyms;
            index -= dataInfoOpInVec.size() + dataInfoOpOutVec.size();
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpProfilingOutVec[index].name());
        }

        auto netIOSym = builderFunc.create<ELF::SymbolOp>(builderFunc.getUnknownLoc(),
                                                          symbolType,  // mlir::Type
                                                          funcArg,     // mlir::Value inputArg
                                                          isBuiltin,   // mlir::UnitAttr
                                                          nameSym,     // mlir::StringAttr
                                                          typeSym,     // vpux::ELF::SymbolTypeAttrAttr
                                                          sizeSym,     // size
                                                          valueSym     // value
        );

        symsVecPtr->push_back(netIOSym.getResult());
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

    // If profiling is enabled add also profiling output symbol table
    if (!dataInfoOpProfilingOutVec.empty()) {
        ELF::CreateSymbolTableSectionOp createProfOutputSymTableSectionOp =
                builderFunc.create<ELF::CreateSymbolTableSectionOp>(
                        mlir::UnknownLoc::get(ctx),
                        vpux::ELF::SectionType::get(ctx),                 // mlir::Type
                        ".symtab.prof_output",                            // llvm::StringRef secName,
                        vpux::ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT,  // vpux::ELF::SectionFlagsAttr secFlags,
                        nullptr                                           // mlir::UnitAttr
                );
        //
        mlir::Region& regProfOutputSymTabSec = createProfOutputSymTableSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkProfOutputSymTabSec = new mlir::Block();
        //
        // This instruction has to be before defining builderSymTabSec to avoid SegFault
        regProfOutputSymTabSec.push_back(blkProfOutputSymTabSec);
        //
        mlir::OpBuilder builderProfOutputSymTabSec(blkProfOutputSymTabSec, blkProfOutputSymTabSec->begin());
        profOutputSymTabValue = createProfOutputSymTableSectionOp.getResult();

        for (auto profOutputSym : profOutputSyms) {
            builderProfOutputSymTabSec.create<ELF::PutOpInSectionOp>(
                    builderProfOutputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                    profOutputSym                                // mlir::Value inputArg
            );
        }
    }
}

void ConvertVPUMI37XX2ELFPass::createKernelParamsRelocs(mlir::func::FuncOp func) {
    auto kernelParamsOps = func.getOps<vpux::VPUMI37XX::KernelParamsOp>();

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

    mlir::Value kernelParamsSectionValue = targetSection.getOperation()->getResult(0);
    auto paramsAutoRelocBuilder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    for (auto kernelParamsOp : kernelParamsOps) {
        auto kernelParamsOpVal = kernelParamsOp.getResult();
        auto partial_addend = ELF::getOffsetOfOpInSection(kernelParamsOpVal, kernelParamsSectionValue) +
                              kernelParamsOp.getParamsStructSize();

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

            partial_addend += sizeof(int64_t) * getMemStrides(kernelOutputsIt.value()).size();
        }
    }
}

mlir::Value ConvertVPUMI37XX2ELFPass::lookupELFSymbol(mlir::Value& symtabValue, mlir::Value& sym_input_value) {
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

void ConvertVPUMI37XX2ELFPass::createActKernelRelocs(mlir::func::FuncOp func) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTab;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;

    // range relocs
    auto actKernelRangeOps = func.getOps<vpux::VPUMI37XX::ActKernelRangeOp>();
    for (auto actKernelRangeOp : actKernelRangeOps) {
        targetSection = relocationManager.getSection(actKernelRangeOp.getResult());

        auto kernelText = actKernelRangeOp.kernel_text_index();

        symTab = relocationManager.getSymTab(kernelText);

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelTextSection"].getDefiningOp());

        auto kernelTextSection = relocationManager.getSection(kernelText).getOperation()->getResult(0);

        builder.create<ELF::RelocOp>(kernelText.getLoc(), actKernelRangeOp, kernelText,
                                     vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym,
                                     ELF::getOffsetOfOpInSection(kernelText, kernelTextSection));
    }

    // invo relocs
    auto actKernelInvoOps = func.getOps<vpux::VPUMI37XX::ActKernelInvocationOp>();

    for (auto actKernelInvoOp : actKernelInvoOps) {
        auto actKernelInvoOpIndex = actKernelInvoOp.index().getType().cast<vpux::VPURegMapped::IndexType>();
        auto associatedRangeOp =
                llvm::dyn_cast<vpux::VPUMI37XX::ActKernelRangeOp>(actKernelInvoOp.range_index().getDefiningOp());

        targetSection = relocationManager.getSection(actKernelInvoOp.getResult());

        // range reloc
        symTab = relocationManager.getCMXSymTab();

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        builder.create<ELF::RelocOp>(
                relocSection.getLoc(), actKernelInvoOp.getResult(), associatedRangeOp.getResult(),
                vpux::ELF::RelocationTypeAttr::R_VPU_32_RTM,
                elfCMXMappingSyms[static_cast<int>(vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_ACT)].getResult(),
                sizeof(nn_public::VpuActKernelRange));

        // data section reloc
        auto kernelData = associatedRangeOp.kernel_args_index();

        symTab = relocationManager.getSymTab(kernelData);

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelDataSection"].getDefiningOp());

        auto kernelDataSection = relocationManager.getSection(kernelData).getOperation()->getResult(0);

        builder.create<ELF::RelocImmOffsetOp>(actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                                              offsetof(nn_public::VpuActKernelInvocation, data_window_base),
                                              vpux::ELF::RelocationTypeAttr::R_VPU_32, sourceSym,
                                              ELF::getOffsetOfOpInSection(kernelData, kernelDataSection));

        // params reloc
        mlir::Value associatedKernelParamsOp;
        auto kernelParamsOps = func.getOps<vpux::VPUMI37XX::KernelParamsOp>();
        for (auto kernelParamsOp : kernelParamsOps) {
            auto kernelParamsOpIndex = kernelParamsOp.index().getType().cast<vpux::VPURegMapped::IndexType>();
            if (kernelParamsOpIndex.getValue() == actKernelInvoOpIndex.getValue()) {
                associatedKernelParamsOp = kernelParamsOp.getResult();
                break;
            }
        }

        symTab = relocationManager.getSymTab(associatedKernelParamsOp);

        relocSection = relocationManager.getRelocSection(targetSection, symTab);

        builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        sourceSym = mlir::dyn_cast<ELF::SymbolOp>(symbolMap["sym_kernelParamsSection"].getDefiningOp());

        auto kernelParamsSection = relocationManager.getSection(associatedKernelParamsOp).getOperation()->getResult(0);

        builder.create<ELF::RelocImmOffsetOp>(
                actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                offsetof(nn_public::VpuActKernelInvocation, kernel_args), vpux::ELF::RelocationTypeAttr::R_VPU_32,
                sourceSym, ELF::getOffsetOfOpInSection(associatedKernelParamsOp, kernelParamsSection));

        // profiling reloc
        // perf_packet_out field from ActKernelInvocation structure needs to point to
        // profiling buffer allocated by compiler
        if (auto profBuffer = actKernelInvoOp.profiling_data()) {
            symTab = relocationManager.getCMXSymTab();

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            auto kernelProfilingBinaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(profBuffer.getDefiningOp());

            size_t addend = 0;

            if (kernelProfilingBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELF::getOffsetOfOpInSection(profBuffer);
            } else {
                auto kernelProfilingSection = relocationManager.getSection(profBuffer);
                sourceSym = ELF::RelocationManager::getSymbol(kernelProfilingSection);
                mlir::Value kernelProfilingSectionValue = kernelProfilingSection.getOperation()->getResult(0);
                addend = ELF::getOffsetOfOpInSection(profBuffer, kernelProfilingSectionValue);
            }

            builder.create<ELF::RelocImmOffsetOp>(actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                                                  offsetof(nn_public::VpuActKernelInvocation, perf_packet_out),
                                                  vpux::ELF::RelocationTypeAttr::R_VPU_32_SUM, sourceSym, addend);
        }
    }
}

void ConvertVPUMI37XX2ELFPass::setupActKernelRtConfigs(mlir::func::FuncOp func, mlir::ModuleOp moduleOp,
                                                       mlir::MLIRContext* ctx) {
    auto mappedInferenceOps = func.getOps<VPUMI37XX::MappedInferenceOp>();

    VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");

    auto mappedInferenceOp = *(mappedInferenceOps.begin());

    if (mappedInferenceOp.actKernelInvocationsCount() == 0) {
        return;
    }

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());
    builderFunc.setInsertionPoint(mappedInferenceOp.getOperation());

    auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");

    auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::func::FuncOp>("runtime");

    mlir::Value actShaveRt;
    ELF::ElfSectionInterface actKRtConfigSec;

    auto actShaveStackMemrefType =
            vpux::getLinearMemrefType(ctx, ACT_SHAVE_STACK_SIZE, vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    llvm::SmallVector<mlir::Value, nn_public::VPU_AS_TOTAL> shaveStacks;
    llvm::SmallVector<mlir::Value, nn_public::VPU_AS_TOTAL> shaveStacksSyms;

    for (size_t i = 0; i < nn_public::VPU_AS_TOTAL; i++) {
        auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                          actShaveStackMemrefType,    // Type
                                                                          VPURT::BufferSection::DDR,  // Buffer Type
                                                                          0                           // byteOffset
        );

        auto shaveStackBufferVal = declareBufferOp.getResult();

        shaveStacks.push_back(shaveStackBufferVal);

        auto strIndex = std::to_string(i);
        auto shaveStackSection = builderFunc.create<ELF::CreateLogicalSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),                     // mlir::Type
                std::string(".bss.actShaveStack_").append(strIndex),  // llvm::StringRef secName,
                vpux::ELF::SectionTypeAttr::SHT_NOBITS,               // vpux::ELF::SectionTypeAttr secType,
                vpux::ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE,      // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                             // int64_t secInfo,
                ELF::VPUX_SHAVE_ALIGNMENT                             // int64_t secAddrAlign
        );

        auto builderShaveStackSection = mlir::OpBuilder::atBlockEnd(shaveStackSection.getBlock());

        builderShaveStackSection.create<ELF::PutOpInSectionOp>(
                builderShaveStackSection.getUnknownLoc(),  // endOp->getLoc(),
                shaveStackBufferVal                        // mlir::Value inputArg
        );

        auto actShaveStackSymOp = builderFunc.create<ELF::SymbolOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SymbolType::get(ctx),                                                 // mlir::Type
                shaveStackSection.getResult(),                                                   // mlir::Value inputArg
                isBuiltin,                                                                       // mlir::UnitAttr
                mlir::StringAttr::get(ctx, std::string("sym_actShaveStack_").append(strIndex)),  // mlir::StringAttr
                typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                sizeSym,  // size
                valueSym  // value
        );
        symbolMap[std::string("sym_actShaveStack_").append(strIndex)] = actShaveStackSymOp.getResult();
        shaveStacksSyms.push_back(actShaveStackSymOp.getResult());
    }

    mappedInferenceOp.actShaveStacksMutable().assign(mlir::ValueRange(shaveStacks));

    if (runtimeKernelFunction) {
        const auto kernelElf =
                std::string(runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code").getValue());

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        auto actShvRtOp = builderFunc.create<VPUMI37XX::ActShaveRtOp>(builderFunc.getUnknownLoc(), trivialIndexType,
                                                                      mlir::StringAttr::get(ctx, kernelElf));

        actShaveRt = actShvRtOp.getResult();

        actKRtConfigSec = builderFunc.create<ELF::CreateSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),          // mlir::Type
                ".text.actKernelRtConfigSec",              // llvm::StringRef secName,
                vpux::ELF::SectionTypeAttr::SHT_PROGBITS,  // vpux::ELF::SectionTypeAttr secType,
                vpux::ELF::SectionFlagsAttr::SHF_NONE,     // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                  // int64_t secInfo,
                1024                                       // int64_t secAddrAlign
        );

        mappedInferenceOp.actShaveRtMutable().assign(actShaveRt);

    } else {
        auto actRtCodeBufferMemrefType =
                vpux::getLinearMemrefType(ctx, ACT_RT_CODE_BUFFER_SIZE, vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

        auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                          actRtCodeBufferMemrefType,  // Type
                                                                          VPURT::BufferSection::DDR,  // Buffer Type
                                                                          0                           // byteOffset
        );

        actShaveRt = declareBufferOp.getResult();

        actKRtConfigSec = builderFunc.create<ELF::CreateLogicalSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),        // mlir::Type
                ".bss.actKernelRtConfigSec",             // llvm::StringRef secName,
                vpux::ELF::SectionTypeAttr::SHT_NOBITS,  // vpux::ELF::SectionTypeAttr secType,
                vpux::ELF::SectionFlagsAttr::SHF_NONE,   // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                // int64_t secInfo,
                ELF::VPUX_SHAVE_ALIGNMENT                // int64_t secAddrAlign
        );
    }

    // Depending on the case, the Section must be binary or logical
    // Refactor such that it comprises both logic

    auto builderElfSectionOpReg = mlir::OpBuilder::atBlockEnd(actKRtConfigSec.getBlock());

    builderElfSectionOpReg.create<ELF::PutOpInSectionOp>(builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                                                         actShaveRt  // mlir::Value inputArg
    );

    mlir::Value actKRtConfigSecValue = actKRtConfigSec.getOperation()->getResult(0);

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

    for (auto shaveStackSym : shaveStacksSyms) {
        builderActKRtConfigSymTab.create<ELF::PutOpInSectionOp>(builderActKRtConfigSymTab.getUnknownLoc(),
                                                                shaveStackSym);
    }

    mlir::Value actKRtConfigSymValue = actKRtConfigSym.getResult();

    ELF::ElfSectionInterface mappedInferenceSec =
            mlir::cast<ELF::ElfSectionInterface>(mappedInferenceSectionOpValue.getDefiningOp());

    VPUX_THROW_UNLESS(mappedInferenceSec != nullptr, "CreateActKernelConfig: MappedInference section is null");

    auto relocSection = relocationManager.getRelocSection(mappedInferenceSec, actKRtConfigSymTab);

    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    for (auto mappedInferenceOp : mappedInferenceOps) {
        builderMappedInfRelocSec.create<ELF::RelocImmOffsetOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
                        offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base),
                vpux::ELF::RelocationTypeAttr::R_VPU_32, actKRtConfigSymValue, 0);

        for (auto shaveStack : shaveStacks | indexed) {
            const auto index = shaveStack.index();
            builderMappedInfRelocSec.create<ELF::RelocOp>(
                    builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), shaveStack.value(),
                    vpux::ELF::RelocationTypeAttr::R_VPU_32, shaveStacksSyms[index], ACT_SHAVE_STACK_SIZE);
        }
    }
}

void ConvertVPUMI37XX2ELFPass::createDPURelocs(mlir::func::FuncOp func) {
    auto invariants = func.getOps<VPUMI37XX::DPUInvariantOp>();

    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTabOfInput;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;
    VPURT::DeclareBufferOp declarator;

    ELF::SymbolOp weightTableStartSym;
    uint64_t weightTableStartAddend = 0;

    vpux::IndexedSymbolAttr bufferMemSpace;
    llvm::Optional<VPURT::BufferSection> bufferSection;

    // TODO: E#54007 currently ignoring sparsity and SOH/SOK.
    for (auto invariant : invariants) {
        int64_t addend = 0;

        auto opType = invariant.task_type();

        auto result = invariant.index();
        targetSection = relocationManager.getSection(result);

        // inputs resolution
        auto input = invariant.input();
        declarator = mlir::cast<VPURT::DeclareBufferOp>(input.getDefiningOp());

        symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                : relocationManager.getSymTab(input);

        relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        addend = declarator.byteOffset();

        bufferMemSpace = input.getType().cast<vpux::NDTypeInterface>().getMemSpace();
        bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
        VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

        if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
            sourceSym = elfCMXMappingSyms[static_cast<int>(
                    vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
            // if we DO NOT have segmentation we relocate to Start_of_CMX + (sectionIdx * SLICE_LENGHT) + local_offset
            // if we DO have segmentation we relocate to start_of_CMX + local_offset. Distributed buffers always
            // assume we start from cluster 0
            if (!invariant.is_segmented().value_or(false) || invariant.task_type() == VPUIP::NCETaskType::ELTWISE) {
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            }
        } else {
            sourceSym = relocationManager.getSymbol(targetSection);
        }

        auto regsOffset = offsetof(nn_public::VpuDPUInvariant, registers_);

        // input relocs, relocating act_offset[0] registers
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[0]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[1]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[2]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        builder.create<ELF::RelocImmOffsetOp>(input.getLoc(), invariant,
                                              regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[3]),
                                              ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);

        if (auto inputSparsityMap = invariant.input_sparsity_map()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(inputSparsityMap.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.byteOffset();

            bufferMemSpace = inputSparsityMap.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

            if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            bool isSegmented = invariant.is_segmented().value_or(false);
            bool isEltwise = invariant.task_type() == VPUIP::NCETaskType::ELTWISE;

            auto secIdx = bufferMemSpace.getIndex().value_or(0);
            addend += (isSegmented && !isEltwise) ? 0 : secIdx * NNCMX_SLICE_SIZE;

            builder.create<ELF::RelocImmOffsetOp>(
                    invariant.input_sparsity_map().getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr) + sizeof(uint32_t),
                    ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);

            if (isEltwise) {
                auto elop_addend = addend;
                if (auto tensorBSparsityMapVal = invariant.weights_sparsity_map()) {
                    auto tensorBSparsity = mlir::cast<VPURT::DeclareBufferOp>(tensorBSparsityMapVal.getDefiningOp());
                    elop_addend = tensorBSparsity.byteOffset();
                }
                builder.create<ELF::RelocImmOffsetOp>(
                        invariant.input_sparsity_map().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, elop_sparsity_addr),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, elop_addend);
            } else if (isSegmented) {
                addend += NNCMX_SLICE_SIZE;
                builder.create<ELF::RelocImmOffsetOp>(
                        invariant.input_sparsity_map().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr[1]) + sizeof(uint32_t),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
            }
        }

        if (auto inputSETable = invariant.input_storage_element_table()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(inputSETable.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);

            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.byteOffset();

            bufferMemSpace = inputSETable.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

            if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            builder.create<ELF::RelocImmOffsetOp>(
                    invariant.input_sparsity_map().getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr),
                    ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        }

        // weights
        if (auto weights = invariant.weights()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            // for weights, we only need to set the start of CMX as base offset. Actual slice_start based offsets of
            // actual weighs are in the weights table
            addend = 0;

            bufferMemSpace = weights.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

            if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            if (opType != VPUIP::NCETaskType::ELTWISE) {
                builder.create<ELF::RelocImmOffsetOp>(
                        weights.getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
            } else {
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp()).byteOffset() +
                                   (secIdx * NNCMX_SLICE_SIZE);

                auto actSecIdx =
                        invariant.input().getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
                auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.input().getDefiningOp()).byteOffset() +
                               (actSecIdx * NNCMX_SLICE_SIZE);

                // correlated with serializer, where rest of the offsets are expected to be directly filled, in
                // accordance with this if-then-else
                builder.create<ELF::RelocImmOffsetOp>(
                        invariant.input().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[0]),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, std::min(actOffs, weightsOffs));
            }
        }

        // no output in case of continued convolution
        if (!invariant.is_continued()) {
            const auto outputBuffs = invariant.output_buffs();
            auto minOutIt =
                    std::min_element(outputBuffs.begin(), outputBuffs.end(), [](mlir::Value lhs, mlir::Value rhs) {
                        auto lhsSliceIdx =
                                lhs.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
                        auto rhsSliceIdx =
                                rhs.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
                        return lhsSliceIdx < rhsSliceIdx;
                    });
            auto output = *minOutIt;

            declarator = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);

            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.byteOffset();

            bufferMemSpace = output.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

            if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            // in case of output the BASE_ADR_0 register is the only one we need to statically relocate. The actual
            // register that the DPU will get is ODU_ACT_BASE. By current contract with shaveNN, it will read BASE_ADR_0
            // reg and write it in ODU_ACT_BASE.  In case of Broadcasting the CAST registers will be configured, each
            // with having a fix offset from the base. The offset will be SLICE_SIZE. This will assume symmetrical
            // allocation and contigious slices. For this logic we will always configure the address of smallest slice
            // index
            static constexpr uint32_t base0Offset = offsetof(nn_public::VpuDPUInvariantRegisters, base_adr[0]);
            static constexpr uint32_t base1Offset = offsetof(nn_public::VpuDPUInvariantRegisters, base_adr[1]);

            builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + base0Offset,
                                                  ELF::RelocationTypeAttr::R_VPU_32_MULTICAST_BASE, sourceSym, addend);
            builder.create<ELF::RelocImmOffsetOp>(output.getLoc(), invariant, regsOffset + base1Offset,
                                                  ELF::RelocationTypeAttr::R_VPU_32_MULTICAST_BASE, sourceSym, addend);

            if (auto outputSparsityMap = invariant.output_sparsity_map_buff()) {
                declarator = mlir::cast<VPURT::DeclareBufferOp>(outputSparsityMap.getDefiningOp());

                symTabOfInput =
                        declarator.section() == VPURT::BufferSection::CMX_NN
                                ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                : relocationManager.getSymTab(input);

                relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

                builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                addend = declarator.byteOffset();

                bufferMemSpace = output.getType().cast<vpux::NDTypeInterface>().getMemSpace();
                bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
                VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

                if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                    sourceSym = elfCMXMappingSyms[static_cast<int>(
                            vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                    auto secIdx = bufferMemSpace.getIndex().value_or(0);
                    addend += secIdx * NNCMX_SLICE_SIZE;
                } else {
                    sourceSym = relocationManager.getSymbol(targetSection);
                }

                builder.create<ELF::RelocImmOffsetOp>(
                        output.getLoc(), invariant, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sp_base),
                        ELF::RelocationTypeAttr::R_VPU_32_MULTICAST_BASE, sourceSym, addend);
            }
        }

        // weights table

        if (auto weightTable = invariant.weight_table()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weightTable.getDefiningOp());

            symTabOfInput = declarator.section() == VPURT::BufferSection::CMX_NN
                                    ? mlir::cast<ELF::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                                    : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = 0;

            bufferMemSpace = weightTable.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

            if (bufferSection.getValue() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            // wt_offset needs to be set even if there is no weights operand in MAXPOOL or AVEPOOL
            if (!invariant.weights() &&
                (opType == VPUIP::NCETaskType::MAXPOOL || opType == VPUIP::NCETaskType::AVEPOOL)) {
                builder.create<ELF::RelocImmOffsetOp>(
                        weightTable.getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                        ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
            }

            addend += declarator.byteOffset();
            weightTableStartSym = sourceSym;
            weightTableStartAddend = addend;
            builder.create<ELF::RelocImmOffsetOp>(
                    weightTable.getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, weight_start),
                    ELF::RelocationTypeAttr::R_VPU_32, sourceSym, addend);
        }

        // variant to invariant relocation
        auto children = invariant.getResult().getUsers();
        for (auto child : children) {
            auto variant = mlir::dyn_cast<VPUMI37XX::DPUVariantOp>(child);
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
                                         sizeof(nn_public::VpuDPUInvariant));

            if (invariant.weight_table()) {
                builder.create<ELF::RelocImmOffsetOp>(
                        variant.getLoc(), variant, offsetof(nn_public::VpuDPUVariant, weight_table_offset_),
                        ELF::RelocationTypeAttr::R_VPU_32_SUM, weightTableStartSym, weightTableStartAddend);
            }

            // in case of invariant, the input drives the specific cluster dispatching. We control this based on the
            // variant's cluster_ field .
            bufferMemSpace = invariant.input().getType().cast<vpux::NDTypeInterface>().getMemSpace();
            int64_t clusterIdx = bufferMemSpace.getIndex().value_or(0);

            sourceSym = elfCMXMappingSyms[static_cast<int>(ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_FIFO_BASE)];
            if (clusterIdx) {
                builder.create<ELF::RelocImmOffsetOp>(variant.getLoc(), variant,
                                                      offsetof(nn_public::VpuDPUVariant, cluster_),
                                                      ELF::RelocationTypeAttr::R_VPU_32, sourceSym, clusterIdx);
            }
        }
    }

    return;
}

// }  // namespace

void ConvertVPUMI37XX2ELFPass::createDMARelocs(mlir::func::FuncOp& funcOp, mlir::MLIRContext* ctx,
                                               mlir::SmallVector<int64_t>& dmaCount,
                                               mlir::Operation::operand_range dmaTasks,
                                               mlir::SmallVector<mlir::Value>& dmaSectionValues) {
    ELF::ElfSectionInterface targetSection;
    ELF::CreateSymbolTableSectionOp symTab;
    ELF::CreateRelocationSectionOp relocSection;
    ELF::SymbolOp sourceSym;

    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    for (auto listHead : dmaTasks) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).port();
        auto listElemCount = dmaCount[listIdx];

        ELF::CreateRelocationSectionOp createInputRelocationSectionOp =
                builderFunc.create<ELF::CreateRelocationSectionOp>(
                        mlir::UnknownLoc::get(ctx),
                        vpux::ELF::SectionType::get(ctx),               // mlir::Type
                        ".rlt.DMA_NetInput" + std::to_string(listIdx),  // llvm::StringRef secName,
                        networkInputSymTabValue,                        // sourceSymbolTableSection,
                        dmaSectionValues[listIdx],                      // targetSection,
                        vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                                vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELF::SectionFlagsAttr
                                                                                // secFlags,
                );

        mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkInputRelocSec = new mlir::Block();

        regInputRelocSec.push_back(blkInputRelocSec);

        mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

        ELF::CreateRelocationSectionOp createOutputRelocationSectionOp =
                builderFunc.create<ELF::CreateRelocationSectionOp>(
                        mlir::UnknownLoc::get(ctx),
                        vpux::ELF::SectionType::get(ctx),                // mlir::Type
                        ".rlt.DMA_NetOutput" + std::to_string(listIdx),  // llvm::StringRef secName,
                        networkOutputSymTabValue,                        // sourceSymbolTableSection,
                        dmaSectionValues[listIdx],                       // targetSection,
                        vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                                vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELF::SectionFlagsAttr
                                                                                 // secFlags,
                );
        mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkOutputRelocSec = new mlir::Block();
        regOutputRelocSec.push_back(blkOutputRelocSec);

        mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

        mlir::OpBuilder builderProfOutputRelocSec(ctx);
        if (profOutputSymTabValue) {
            ELF::CreateRelocationSectionOp createProfOutputRelocationSectionOp =
                    builderFunc.create<ELF::CreateRelocationSectionOp>(
                            mlir::UnknownLoc::get(ctx),
                            vpux::ELF::SectionType::get(ctx),                 // mlir::Type
                            ".rlt.DMA_ProfOutput" + std::to_string(listIdx),  // llvm::StringRef secName,
                            profOutputSymTabValue,                            // sourceSymbolTableSection,
                            dmaSectionValues[listIdx],                        // targetSection,
                            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                                    vpux::ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT  // vpux::ELF::SectionFlagsAttr
                                                                                     // secFlags,
                    );
            mlir::Region& regProfOutputRelocSec = createProfOutputRelocationSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkProfOutputRelocSec = new mlir::Block();
            regProfOutputRelocSec.push_back(blkProfOutputRelocSec);

            builderProfOutputRelocSec = mlir::OpBuilder(blkProfOutputRelocSec, blkProfOutputRelocSec->begin());
        }

        targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(dmaSectionValues[listIdx].getDefiningOp());

        // Need to go through individual DMA task lists
        for (auto dmaIdx = 0; dmaIdx < listElemCount; ++dmaIdx) {
            auto dmaOp = mlir::dyn_cast_or_null<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp());
            // input addr
            if (auto dmaInputArg = dmaOp.input().dyn_cast<mlir::BlockArgument>()) {
                if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaInputArg)) {
                    builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
                            vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                            netInputSymValue,                         // ::mlir::Value sourceSymbol
                            0                                         // int64_t addend
                    );
                } else if (mlir::Value netInputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaInputArg)) {
                    builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
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
                                      "Encountered DMA op {} with input {} which has multiple section indexes {}",
                                      dmaOp, dmaInputArg_, funcArgIndex);
                    auto inputOffset = dmaInputArg_.byteOffset();
                    auto funcArg = funcOp.getArgument(funcArgIndex[0]);
                    if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, funcArg)) {
                        builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                                builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
                                vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                                netInputSymValue,                         // ::mlir::Value sourceSymbol
                                inputOffset                               // int64_t addend
                        );
                    }
                } else if (dmaInputArg_ && (dmaInputArg_.getMemorySpace() == VPURT::BufferSection::NetworkOutput)) {
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaInputArg_.sectionIndex().getValue());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with input {} which has multiple section indexes {}",
                                      dmaOp, dmaInputArg_, funcArgIndex);
                    auto inputOffset = dmaInputArg_.byteOffset();
                    auto funcArg =
                            funcOp.getArgument(funcArgIndex[0] + funcOp.getNumArguments() - funcOp.getNumResults());
                    if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, funcArg)) {
                        builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                                builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
                                vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                                netOutputSymValue,                        // ::mlir::Value sourceSymbol
                                inputOffset                               // int64_t addend
                        );
                    }
                } else {
                    auto dmaInput = dmaOp.input();

                    symTab = relocationManager.getSymTab(dmaInput);

                    relocSection = relocationManager.getRelocSection(targetSection, symTab);

                    auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                    auto inputMemSpace = dmaInput.getType().cast<vpux::NDTypeInterface>().getMemSpace();
                    // if no bufferSection specified explicitly, use the Defaul DDR
                    auto bufferSection = inputMemSpace ? VPURT::symbolizeBufferSection(inputMemSpace.getLeafName())
                                                       : VPURT::BufferSection::DDR;
                    VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

                    size_t addend = 0;

                    if (bufferSection == VPURT::BufferSection::CMX_NN) {
                        sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                        addend = ELF::getOffsetOfOpInSection(dmaInput);
                    } else if (bufferSection == VPURT::BufferSection::Register) {
                        sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_HW_REGISTER)];
                        addend = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(dmaInput.getDefiningOp()).byteOffset();
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
            auto outputBuffs = dmaOp.output_buffs();

            if (auto dmaOutputArg = outputBuffs[0].dyn_cast<mlir::BlockArgument>()) {
                VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as blockArgument with multiple outputs");
                if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, dmaOutputArg)) {
                    builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                            vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                            netOutputSymValue,                        // ::mlir::Value sourceSymbol
                            0                                         // int64_t addend
                    );
                } else if (mlir::Value netOutputSymValue = lookupELFSymbol(networkInputSymTabValue, dmaOutputArg)) {
                    builderInputRelocSec.create<ELF::RelocImmOffsetOp>(
                            builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                            offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                            vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                            netOutputSymValue,                        // ::mlir::Value sourceSymbol
                            0                                         // int64_t addend
                    );
                }
            } else {
                auto dmaOutputArg_ = outputBuffs[0].getDefiningOp<VPURT::DeclareBufferOp>();
                VPUX_THROW_UNLESS(dmaOutputArg_,
                                  "Encountered DMA op {} with output {} which is neither mlir::BlockArgument, nor "
                                  "VPURT::DeclareBufferOp",
                                  dmaOp, dmaOutputArg_);

                if (dmaOutputArg_.getMemorySpace() == VPURT::BufferSection::NetworkOutput) {
                    VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as NetworkOut with multiple outputs");
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaOutputArg_.sectionIndex().getValue());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with output {} which has multiple secion indexes {}",
                                      dmaOp, dmaOutputArg_, funcArgIndex);
                    auto outputOffset = dmaOutputArg_.byteOffset();
                    auto funcArg =
                            funcOp.getArgument(funcArgIndex[0] + funcOp.getNumArguments() - funcOp.getNumResults());
                    if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, funcArg)) {
                        builderOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                                builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                                vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                                netOutputSymValue,                        // ::mlir::Value sourceSymbol
                                outputOffset                              // int64_t addend
                        );
                    }
                } else if (dmaOutputArg_.getMemorySpace() == VPURT::BufferSection::ProfilingOutput) {
                    VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as NetworkOut with multiple outputs");
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaOutputArg_.sectionIndex().getValue());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with output {} which has multiple secion indexes {}",
                                      dmaOp, dmaOutputArg_, funcArgIndex);
                    VPUX_THROW_UNLESS(funcArgIndex[0] == 0, "Only profiling output index 0 is supported, got '{0}'",
                                      funcArgIndex[0]);
                    auto outputOffset = dmaOutputArg_.byteOffset();
                    auto funcArg = funcOp.getArgument(funcOp.getNumArguments() - 1);
                    if (mlir::Value profOutputSymValue = lookupELFSymbol(profOutputSymTabValue, funcArg)) {
                        builderProfOutputRelocSec.create<ELF::RelocImmOffsetOp>(
                                builderProfOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                                vpux::ELF::RelocationTypeAttr::R_VPU_64,  // relocationType
                                profOutputSymValue,                       // ::mlir::Value sourceSymbol
                                outputOffset                              // int64_t addend
                        );
                    }
                } else {
                    auto dmaOutput = outputBuffs[0];

                    symTab = relocationManager.getSymTab(dmaOutput);

                    relocSection = relocationManager.getRelocSection(targetSection, symTab);

                    auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                    auto outputMemSpace = dmaOutput.getType().cast<vpux::NDTypeInterface>().getMemSpace();
                    // if no bufferSection specified explicitly, use the Defaul DDR
                    auto bufferSection = outputMemSpace ? VPURT::symbolizeBufferSection(outputMemSpace.getLeafName())
                                                        : VPURT::BufferSection::DDR;
                    VPUX_THROW_UNLESS(bufferSection.hasValue(), "Buffer with no section associated");

                    size_t addend = 0;

                    if (bufferSection == VPURT::BufferSection::CMX_NN) {
                        sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                        addend = ELF::getOffsetOfOpInSection(dmaOutput);
                    } else if (bufferSection == VPURT::BufferSection::Register) {
                        sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_HW_REGISTER)];
                        addend = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(dmaOutput.getDefiningOp()).byteOffset();
                    } else {
                        auto dmaOutputSection = relocationManager.getSection(dmaOutput);
                        sourceSym = ELF::RelocationManager::getSymbol(dmaOutputSection);
                        mlir::Value dmaOutputSectionValue = dmaOutputSection.getOperation()->getResult(0);
                        addend = ELF::getOffsetOfOpInSection(dmaOutput, dmaOutputSectionValue);
                    }

                    // in case of broadcast output using OR relocation. The DST will have the default MASK value for
                    // multicast;
                    auto relocType = outputBuffs.size() > 1 ? vpux::ELF::RelocationTypeAttr::R_VPU_64_OR
                                                            : vpux::ELF::RelocationTypeAttr::R_VPU_64;
                    builder.create<ELF::RelocOp>(dmaOutput.getLoc(), dmaOp, dmaOutput, relocType, sourceSym, addend);
                }
            }

            // link_address
            if (static_cast<uint32_t>(listElemCount) > dmaOp.getType().getValue() + 1) {
                symTab = relocationManager.getCMXSymTab();

                relocSection = relocationManager.getRelocSection(targetSection, symTab);

                auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                builder.create<ELF::RelocImmOffsetOp>(
                        relocSection.getLoc(), dmaOp, offsetof(nn_public::VpuDMATask, transaction_),
                        vpux::ELF::RelocationTypeAttr::R_VPU_32_RTM,
                        elfCMXMappingSyms[static_cast<int>(vpux::ELF::CMXMappingSymbolAttr::VPU_NNRD_SYM_RTM_DMA0) +
                                          listIdx]
                                .getResult(),
                        sizeof(nn_public::VpuDMATask));
            }

            listHead = getNextDMATask(listHead);
        }
    }
}

void ConvertVPUMI37XX2ELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::func::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    _log.trace("ConvertVPUMI37XX2ELFPass::safeRunOnFunc(): START\n {0}\n", moduleOp);

    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

    relocationManager.init(funcOp);

    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));
    auto mappedInferenceOps = funcOp.getOps<VPUMI37XX::MappedInferenceOp>();
    VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");
    auto mappedInferenceOp = *(mappedInferenceOps.begin());

    auto dmaCount = parseIntArrayAttr<int64_t>(mappedInferenceOp.dmaCount());
    auto barrierCount = mappedInferenceOp.barrierCount();
    auto rangeCount = mappedInferenceOp.actKernelRangesCount();
    auto invoCount = mappedInferenceOp.actKernelInvocationsCount();
    auto invariantCount = mappedInferenceOp.invariantCount();
    auto variantCount = mappedInferenceOp.variantCount();

    auto dmaTasks = mappedInferenceOp.dmaTasks();
    auto barrierTasks = mappedInferenceOp.barrierTasks();
    auto actKernelRanges = mappedInferenceOp.actKernelRanges();
    auto actKernelInvocations = mappedInferenceOp.actKernelInvocations();
    auto invariantTasks = mappedInferenceOp.invariantTasks();
    auto variantTasks = mappedInferenceOp.variantTasks();

    auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

    //
    // Sections Creation
    //

    mlir::SmallVector<mlir::Value> nndmaSectionOpValues;

    nndmaSectionOpValues = createDMASections(funcOp, ctx, dmaCount, dmaTasks);

    mlir::Value barrierSectionOpValue = createSection<vpux::VPUMI37XX::ConfigureBarrierOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.BarrierConfigs", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelTextSectionOpValue = createSection<vpux::VPUMI37XX::DeclareKernelTextOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.KernelText", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelDataSectionOpValue = createSection<vpux::VPUMI37XX::DeclareKernelArgsOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.KernelData", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value kernelParamsSectionOpValue = createSection<vpux::VPUMI37XX::KernelParamsOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.KernelParams", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value actKernelRangesSectionOpValue = createSection<vpux::VPUMI37XX::ActKernelRangeOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.ActKernelRanges", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value actKernelInvosSectionOpValue =
            createSection<vpux::VPUMI37XX::ActKernelInvocationOp, ELF::CreateSectionOp>(
                    funcOp, ctx, ".text.ActKernelInvocations", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mappedInferenceSectionOpValue = createSection<vpux::VPUMI37XX::MappedInferenceOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.MappedInference", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value invariantsSection = createSection<vpux::VPUMI37XX::DPUInvariantOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.DPUInvariants", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    mlir::Value variantsSection = createSection<vpux::VPUMI37XX::DPUVariantOp, ELF::CreateSectionOp>(
            funcOp, ctx, ".text.DPUVariants", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_NONE);

    auto metadataSectionOp = builderFunc.create<ELF::CreateMetadataSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELF::SectionType::get(ctx),                               // mlir::Type
            ".metadata",                                                    // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE,                          // vpux::ELF::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                                       // int64_t secInfo,
            vpux::VPUMI37XX::NetworkMetadataOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto builderMetadataSec = mlir::OpBuilder::atBlockEnd(metadataSectionOp.getBlock());

    builderMetadataSec.create<VPUMI37XX::NetworkMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType);

    if (!cnnOp.getProfilingOutputsInfo().empty()) {
        auto profilingSectionOp = builderFunc.create<ELF::CreateProfilingSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELF::SectionType::get(ctx),       // mlir::Type
                ".profiling",                           // llvm::StringRef secName,
                vpux::ELF::SectionFlagsAttr::SHF_NONE,  // vpux::ELF::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,               // int64_t secInfo,
                vpux::VPUMI37XX::ProfilingMetadataOp::getAlignmentRequirements()  // int64_t secAddrAlign
        );

        auto builderProfilingSec = mlir::OpBuilder::atBlockEnd(profilingSectionOp.getBlock());
        builderProfilingSec.create<VPUMI37XX::ProfilingMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType);
    }

    _log.trace("ConvertVPUMI37XX2ELFPass, after sections creation:\n {0} \n", moduleOp);

    //
    // Create Symbols for the relevant sections
    //

    vpux::ELF::SymbolTypeAttrAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;
    mlir::SmallVector<vpux::ELF::SymbolOp> dmaSectionSyms;

    for (size_t listIdx = 0; listIdx < dmaCount.size(); ++listIdx) {
        dmaSectionSyms.push_back(builderFunc.create<ELF::SymbolOp>(
                mlir::UnknownLoc::get(ctx),
                vpux::ELF::SymbolType::get(ctx),                                         // mlir::Type
                nndmaSectionOpValues[listIdx],                                           // mlir::Value inputArg
                isBuiltin,                                                               // mlir::UnitAttr
                mlir::StringAttr::get(ctx, "sym_dmaSection" + std::to_string(listIdx)),  // mlir::StringAttr
                typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                sizeSym,  // size
                valueSym  // value
                ));

        symbolMap["sym_dmaSection" + std::to_string(listIdx)] = dmaSectionSyms[listIdx].getResult();
    }

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

    for (auto listHead : dmaTasks) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).port();
        builderTasksSymTab.create<ELF::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                         dmaSectionSyms[listIdx].getResult());
    }
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

    _log.trace("ConvertVPUMI37XX2ELFPass, after symtabs creation:\n {0} \n", moduleOp);

    //
    // create general relocs for the tasks
    //

    createDMARelocs(funcOp, ctx, dmaCount, dmaTasks, nndmaSectionOpValues);
    _log.trace("ConvertVPUMI37XX2ELFPass, after DMA Relocs creation:\n {0} \n", moduleOp);

    createKernelParamsRelocs(funcOp);
    createActKernelRelocs(funcOp);
    setupActKernelRtConfigs(funcOp, moduleOp, ctx);
    _log.trace("ConvertVPUMI37XX2ELFPass, after Shave Relocs creation:\n {0} \n", moduleOp);

    createDPURelocs(funcOp);
    _log.trace("ConvertVPUMI37XX2ELFPass, after ActKernel Relocs creation:\n {0} \n", moduleOp);

    //
    // create relocs for the tasks in MappedInference
    //

    ELF::ElfSectionInterface targetSection =
            mlir::dyn_cast<ELF::ElfSectionInterface>(mappedInferenceSectionOpValue.getDefiningOp());
    ELF::CreateSymbolTableSectionOp symTab = tasksSymTabOp;
    ELF::CreateRelocationSectionOp relocSection = relocationManager.getRelocSection(targetSection, symTab);

    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    // Refresh range after mapped inference was updated
    dmaTasks = mappedInferenceOp.dmaTasks();
    for (auto listHead : dmaTasks) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).port();
        builderMappedInfRelocSec.create<ELF::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(), listHead,
                vpux::ELF::RelocationTypeAttr::R_VPU_64, dmaSectionSyms[listIdx].getResult(), 0);
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

    _log.trace("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): FINISH\n {0}\n", moduleOp);
}
}  // namespace

//
// createConvertVPUMI37XX2ELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI37XX2ELFPass(Logger log) {
    return std::make_unique<ConvertVPUMI37XX2ELFPass>(log);
}
