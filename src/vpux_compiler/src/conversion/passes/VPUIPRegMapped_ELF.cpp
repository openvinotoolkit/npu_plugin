//
// Copyright (C) 2022 Intel Corporation.
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

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

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
    template <typename DerivedOpType>
    mlir::Value createSection(mlir::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr,
                              vpux::ELF::SectionTypeAttr secType, vpux::ELF::SectionFlagsAttr secFlags,
                              bool isNNDMAOp = false);

    bool checkIfValueIsNetArg(mlir::Value val, std::string& strRes);
    void createRelocationSection(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& nndmaSectionOp);

    void safeRunOnModule() final;

    Logger _log;

    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

    unsigned int numNndmaOps = 0;
    std::vector<mlir::Value> nndmaOpInput;
    std::vector<mlir::Value> nndmaOpOutput;

    int sizeOfNNDMAOpStruct;
    int offsetOfSrcInNNDMAOpStruct;
    int offsetOfDstInNNDMAOpStruct;
};

// createSection() creates an ELF::CreateSectionOp and puts into its body
//   for each object of type DerivedOpType from the FuncOp func an
//   ELF.PutOpInSectionOp instruction.
//  In case isNNDMAOp is true, then we store in the nndmaOpInput and nndmaOpOutput
//    the input, respectively the output of the current NNDMAOp operation.
template <typename DerivedOpType>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                            std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                            vpux::ELF::SectionFlagsAttr secFlags, bool isNNDMAOp) {
    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    auto elfCreateSectionOp =
            builderFunc.create<ELF::CreateSectionOp>(mlir::UnknownLoc::get(ctx),
                                                     sectionType,  // mlir::Type
                                                     secNameStr,   // llvm::StringRef secName,
                                                     secType,      // vpux::ELF::SectionTypeAttr secType,
                                                     secFlags,     // vpux::ELF::SectionFlagsAttr secFlags,
                                                     1,            // int64_t secInfo,
                                                     64            // int64_t secAddrAlign
            );

    mlir::Block* blkNew = &(elfCreateSectionOp.aRegion().emplaceBlock());

    for (auto op : func.getOps<DerivedOpType>()) {
        if (isNNDMAOp) {
            // isNNDMAOp is true only when DerivedOpType is vpux::VPUIPRegMapped::NNDMAOp
            nndmaOpInput.push_back(((vpux::VPUIPRegMapped::NNDMAOp)op).input());
            nndmaOpOutput.push_back(((vpux::VPUIPRegMapped::NNDMAOp)op).output_buff());
            _log.info("createSection(): nndmaOpInput[numNndmaOps] = {0}\n", nndmaOpInput[numNndmaOps]);
            _log.info("createSection(): nndmaOpOutput[numNndmaOps] = {0}\n", nndmaOpOutput[numNndmaOps]);

            numNndmaOps++;
        }

        mlir::OpBuilder builderElfSectionOpReg(blkNew, blkNew->begin());

        _log.info("createSection(): Before builderElfSectionOpReg.create()\n");

        auto elfPutOpInSectionOp = builderElfSectionOpReg.create<ELF::PutOpInSectionOp>(
                builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                op.getOperation()->getResult(0)          // mlir::Value inputArg
        );

        VPUX_UNUSED(elfPutOpInSectionOp);
    }

    return elfCreateSectionOp.getOperation()->getResult(0);
}

// We check if val is a Neural-Network argument, defined in the IE.CNNNetwork,
//   at operations inputsInfo or outputsInfo, and returns true if so, false
//   otherwise.
bool Convert2VPUIPRegMappedAndELFPass::checkIfValueIsNetArg(mlir::Value val, std::string& strRes) {
    bool res;

    mlir::BlockArgument blockArg = val.dyn_cast_or_null<mlir::BlockArgument>();
    if (blockArg) {
        res = true;

        unsigned int blockArgNum = blockArg.getArgNumber();

        _log.info("    blockArgNum = {0}\n", blockArgNum);

        // By convention, the arguments of the FuncOp are described in-order in
        //   the IE.CNNNetwork block, with its parameters inputsInfo and
        //   outputsInfo of the DataInfo operations.
        //  Note: we should be guaranteed that blockArgNum < diOpInVec.size() + diOpOutVec.size() .
        vpux::IE::DataInfoOp respectiveNetArg;
        if (blockArgNum < diOpInVec.size()) {
            respectiveNetArg = diOpInVec[blockArgNum];
        } else {
            respectiveNetArg = diOpOutVec[blockArgNum - diOpInVec.size()];
        }

        _log.info("    respectiveNetArg = {0}\n", respectiveNetArg);

        strRes = respectiveNetArg.name().str();
    } else {
        res = false;
    }

    return res;
}

// We create for all NNDMAOps, associated ELF.SmybolOps.
//   Then we create symbol tables (of 3 types: standard, for inputs, and for
//   outputs) and populate them with PutAnyInSectionOp. Then we create similarly
//   relocation sections with nndmaSectionOpValue as target section,
//   which we later populate with RelocOps.
void Convert2VPUIPRegMappedAndELFPass::createRelocationSection(mlir::FuncOp funcOp, mlir::MLIRContext* ctx,
                                                               mlir::Value& nndmaSectionOpValue) {
    _log.info("createRelocationSection(): funcOp = {0}\n", funcOp);

    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    // Creating the required ELF.SymbolOps
    std::vector<ELF::SymbolOp> elfSymbolOp;
    unsigned int numElfSymbolOps = 0;

    unsigned int idx;
    for (idx = 0; idx < numNndmaOps; idx++) {
        std::string nameSymStr = "nndmaOp";
        nameSymStr += std::to_string(idx);

        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        mlir::StringAttr nameSym;

        nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");

        // Note: We don't initialize the following 3 vars because they are defined as optional.
        vpux::ELF::SymbolTypeAttrAttr typeSym;
        mlir::IntegerAttr sizeSym;
        mlir::IntegerAttr valueSym;

        std::string tmpStr;

        _log.info("nndmaOpInput[idx] = {0}\n", nndmaOpInput[idx]);

        // We handle the nndmaOpInput[idx] value
        bool res = checkIfValueIsNetArg(nndmaOpInput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");
        }
        _log.info("nameSym = {0}\n", nameSym);

        elfSymbolOp.push_back(builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                                                symbolType,         // mlir::Type
                                                                nndmaOpInput[idx],  // mlir::Value inputArg
                                                                nameSym,            // mlir::StringAttr
                                                                typeSym,            // vpux::ELF::SymbolTypeAttrAttr
                                                                sizeSym,            // size
                                                                valueSym            // value
                                                                ));
        _log.info("elfSymbolOp[{0}] = {1}\n", numElfSymbolOps, elfSymbolOp[numElfSymbolOps]);
        numElfSymbolOps++;

        _log.info("nndmaOpOutput[idx] = {0}\n", nndmaOpOutput[idx]);

        // We handle the nndmaOpInput[idx] value
        res = checkIfValueIsNetArg(nndmaOpOutput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_output");
        }
        _log.info("nameSym = {0}\n", nameSym);

        elfSymbolOp.push_back(builderFunc.create<ELF::SymbolOp>(mlir::UnknownLoc::get(ctx),
                                                                symbolType,          // mlir::Type
                                                                nndmaOpOutput[idx],  // mlir::Value inputArg
                                                                nameSym,             // mlir::StringAttr
                                                                typeSym,             // vpux::ELF::SymbolTypeAttrAttr
                                                                sizeSym,             // size
                                                                valueSym             // value
                                                                ));
        _log.info("elfSymbolOp[{0}] = {1}\n", numElfSymbolOps, elfSymbolOp[numElfSymbolOps]);
        numElfSymbolOps++;
    }

    // Now, we create the symbol tables.
    // First we create the symbol table for the "rest"/normal symbols.
    vpux::ELF::SectionType sectionType2 = vpux::ELF::SectionType::get(ctx);

    ELF::CreateSymbolTableSectionOp createRestSymTableSectionOp =
            builderFunc.create<ELF::CreateSymbolTableSectionOp>(mlir::UnknownLoc::get(ctx),
                                                                sectionType2,                // mlir::Type
                                                                ".rest.symbolTableSection",  // llvm::StringRef secName,
                                                                vpux::ELF::SectionFlagsAttr::SHF_NONE);
    //
    mlir::Region& regRestSymTabSec = createRestSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkRestSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderRestSymTabSec to avoid SegFault
    regRestSymTabSec.push_back(blkRestSymTabSec);
    //
    mlir::OpBuilder builderRestSymTabSec(blkRestSymTabSec, blkRestSymTabSec->begin());

    // Secondly we create the symbol table for the input symbols
    ELF::CreateSymbolTableSectionOp createInputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            sectionType2,                                   // mlir::Type
            ".symtab.input",                                // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    mlir::Region& regInputSymTabSec = createInputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regInputSymTabSec.push_back(blkInputSymTabSec);
    //
    mlir::OpBuilder builderInputSymTabSec(blkInputSymTabSec, blkInputSymTabSec->begin());

    // Thirdly we create the symbol table for the output symbols
    ELF::CreateSymbolTableSectionOp createOutputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            sectionType2,                                    // mlir::Type
            ".symtab.output",                                // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    mlir::Region& regOutputSymTabSec = createOutputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regOutputSymTabSec.push_back(blkOutputSymTabSec);
    //
    mlir::OpBuilder builderOutputSymTabSec(blkOutputSymTabSec, blkOutputSymTabSec->begin());

    // We now create PutOpInSectionOp and put them accordingly with SymbolOps
    //   in the 3 corresponding symbol tables.
    for (idx = 0; idx < numElfSymbolOps; idx++) {
        ELF::PutOpInSectionOp putOpInSectionOp;

        mlir::Value inputArgElfSymbolOp = elfSymbolOp[idx].inputArg();
        mlir::BlockArgument blockArg = inputArgElfSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        _log.info("inputArgElfSymbolOp = {0}\n", inputArgElfSymbolOp);

        if (blockArg) {
            unsigned int blockArgNum = blockArg.getArgNumber();

            _log.info("    inputArgElfSymbolOp is a BlockArgument and blockArgNum = {0}\n", blockArgNum);

            if (blockArgNum < diOpInVec.size()) {
                putOpInSectionOp = builderInputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderInputSymTabSec.getUnknownLoc(),         // endOp->getLoc(),
                        elfSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            } else {
                putOpInSectionOp = builderOutputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderOutputSymTabSec.getUnknownLoc(),        // endOp->getLoc(),
                        elfSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            }
        } else {
            putOpInSectionOp = builderRestSymTabSec.create<ELF::PutOpInSectionOp>(
                    builderRestSymTabSec.getUnknownLoc(),          // endOp->getLoc(),
                    elfSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
            );
        }

        _log.info("createRelocationSection(): putOpInSectionOp = {0}\n", putOpInSectionOp);
    }

    // vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    // We now create the relocation sections
    // First, we create the "rest" relocation section
    ELF::CreateRelocationSectionOp createRestRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            mlir::UnknownLoc::get(ctx),
            sectionType2,                                              // mlir::Type
            ".rlt.dma",                                                // secName, // llvm::StringRef
            createRestSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                       // targetSection,
            // vpux::ELF::SectionFlagsAttr::SHF_NONE                      // vpux::ELF::SectionFlagsAttr secFlags,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK);
    //
    _log.info("createRelocationSection(): createRestRelocationSectionOp = {0}\n", createRestRelocationSectionOp);
    //
    mlir::Region& regRestRelocSec = createRestRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkRestRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regRestRelocSec.push_back(blkRestRelocSec);
    //
    mlir::OpBuilder builderRestRelocSec(blkRestRelocSec, blkRestRelocSec->begin());

    // Secondly, we create the input relocation section
    ELF::CreateRelocationSectionOp createInputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            mlir::UnknownLoc::get(ctx),
            sectionType2,                                               // mlir::Type
            ".rlt.input",                                               // llvm::StringRef secName,
            createInputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    _log.info("createRelocationSection(): createInputRelocationSectionOp = {0}\n", createInputRelocationSectionOp);
    //
    mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regInputRelocSec.push_back(blkInputRelocSec);
    //
    mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

    // Thirdly, we create the output relocation section
    ELF::CreateRelocationSectionOp createOutputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            mlir::UnknownLoc::get(ctx),
            sectionType2,                                                // mlir::Type
            ".rlt.output",                                               // llvm::StringRef secName,
            createOutputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                         // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    _log.info("createRelocationSection(): createOutputRelocationSectionOp = {0}\n", createOutputRelocationSectionOp);
    //
    mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regOutputRelocSec.push_back(blkOutputRelocSec);
    //
    mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

    // We now create the RelocOps
    for (idx = 0; idx < numElfSymbolOps; idx++) {
        bool isBlkFuncArg = false;

        mlir::Value inputArgElfSymbolOp = elfSymbolOp[idx].inputArg();
        _log.info("inputArgElfSymbolOp = {0}\n", inputArgElfSymbolOp);
        mlir::BlockArgument blockArg = inputArgElfSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        unsigned int blockArgNum;

        if (blockArg) {
            isBlkFuncArg = true;
            blockArgNum = blockArg.getArgNumber();
            _log.info("    blockArgNum = {0}\n", blockArgNum);
        }

        ELF::RelocOp elfRelocOp;

        if (isBlkFuncArg) {
            if (blockArgNum < diOpInVec.size()) {
                elfRelocOp = builderInputRelocSec.create<ELF::RelocOp>(
                        builderInputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? offsetOfSrcInNNDMAOpStruct : offsetOfDstInNNDMAOpStruct) +
                                (sizeOfNNDMAOpStruct *
                                 (idx / 2)),  // offsetTargetField (Note: we treat in a simple manner the calculation of
                                              // offsetTargetField. In a near future PR we will treat in a generic way
                                              // the calculation of offsetTargetField.)
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,        // relocationType
                        elfSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                        0                                               // int64_t addend
                );
            } else {
                elfRelocOp = builderOutputRelocSec.create<ELF::RelocOp>(
                        builderOutputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? offsetOfSrcInNNDMAOpStruct : offsetOfDstInNNDMAOpStruct) +
                                (sizeOfNNDMAOpStruct *
                                 (idx / 2)),  // offsetTargetField (Note: we treat in a simple manner the calculation of
                                              // offsetTargetField. In a near future PR we will treat in a generic way
                                              // the calculation of offsetTargetField.)
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,        // relocationType
                        elfSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                        0                                               // int64_t addend
                );
            }
        } else {
            elfRelocOp = builderRestRelocSec.create<ELF::RelocOp>(
                    builderRestRelocSec.getUnknownLoc(),
                    ((idx & 1) == 0 ? offsetOfSrcInNNDMAOpStruct : offsetOfDstInNNDMAOpStruct) +
                            (sizeOfNNDMAOpStruct *
                             (idx / 2)),  // offsetTargetField (Note: we treat in a simple manner the calculation of
                                          // offsetTargetField. In a near future PR we will treat in a generic way the
                                          // calculation of offsetTargetField.)
                    vpux::ELF::RelocationTypeAttr::R_VPU_64,        // relocationType
                    elfSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                    0                                               // int64_t addend
            );
        }

        _log.info("createRelocationSection(): elfRelocOp = {0}\n", elfRelocOp);
    }

    _log.info("createRelocationSection(): funcOp = {0}\n", funcOp);
}

void Convert2VPUIPRegMappedAndELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    vpux::VPUIPRegMapped::NNDMAOp nndmaOpTmp;
    sizeOfNNDMAOpStruct = (int)nndmaOpTmp.getBinarySize();
    offsetOfSrcInNNDMAOpStruct = offsetof(host_parsing::DmaDescriptor, src);
    offsetOfDstInNNDMAOpStruct = offsetof(host_parsing::DmaDescriptor, dst);
    //
    _log.info("sizeOfNNDMAOpStruct = {0}\n", sizeOfNNDMAOpStruct);
    _log.info("offsetOfSrcInNNDMAOpStruct = {0}\n", offsetOfSrcInNNDMAOpStruct);
    _log.info("offsetOfDstInNNDMAOpStruct = {0}\n", offsetOfDstInNNDMAOpStruct);

    _log.info("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = {0}\n", moduleOp);

    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

    diOpInVec = cnnOp.getInputsInfo();
    diOpOutVec = cnnOp.getOutputsInfo();
    //
    _log.info("  diOpInVec.size() = {0}\n", diOpInVec.size());
    _log.info("  diOpOutVec.size() = {0}\n", diOpOutVec.size());

    // We build 4 different createSections, one for VPUIPRegMapped::DeclareBufferOp,
    //   one for Const::DeclareOp, one for VPUIPRegMapped::NNDMAOp and
    //   one for VPUIPRegMapped::ConfigureBarrierOp.
    createSection<vpux::VPURT::DeclareBufferOp>(funcOp, ctx, ".data.Weights", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
    createSection<vpux::Const::DeclareOp>(funcOp, ctx, ".data.Weights_ct", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                          vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
    mlir::Value nndmaSectionOpValue = createSection<vpux::VPUIPRegMapped::NNDMAOp>(
            funcOp, ctx, ".text.dmaTasks", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELF::SectionFlagsAttr::SHF_ALLOC | vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR, true);
    createSection<vpux::VPUIPRegMapped::ConfigureBarrierOp>(funcOp, ctx, ".text.BarrierConfigs",
                                                            vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                            vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));
    //
    int barrierCount = 0;
    for (auto op : funcOp.getOps<VPUIPRegMapped::ConfigureBarrierOp>()) {
        VPUX_UNUSED(op);
        barrierCount++;
    }
    //
    mlir::Value dmaList;
    mlir::Value invariantList;
    mlir::Value variantList;
    mlir::Value actInvocations;
    mlir::Value barrierList;
    //
    VPUIPRegMapped::MappedInferenceOp mappedInferenceOp =
            builderFunc.create<VPUIPRegMapped::MappedInferenceOp>(mlir::UnknownLoc::get(ctx),
                                                                  dmaList,         // mlir::Value dmaList
                                                                  invariantList,   // mlir::Value invariantList
                                                                  variantList,     // mlir::Value variantList
                                                                  actInvocations,  // mlir::Value actInvocations
                                                                  barrierList,     // mlir::Value barrierList
                                                                  numNndmaOps,     // uint32_t dmaCount
                                                                  0,               // uint32_t invariantCount
                                                                  0,               // uint32_t variantCount
                                                                  0,               // uint32_t actInvocationsCount
                                                                  barrierCount     // uint32_t barrierCount
            );
    VPUX_UNUSED(mappedInferenceOp);
    _log.fatal("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): numNndmaOps = {0}", numNndmaOps);
    _log.fatal("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): barrierCount = {0}", barrierCount);

    // Now, for each NNDMAOp input and output we want to perform relocation
    createRelocationSection(funcOp, ctx, nndmaSectionOpValue);

    _log.info("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = {0}\n", moduleOp);
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
