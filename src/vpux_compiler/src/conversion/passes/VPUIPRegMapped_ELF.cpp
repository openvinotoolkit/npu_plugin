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
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "llvm/Support/Debug.h"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
//#include "mlir/IR/PatternMatch.h"

#include "host_parsed_inference.h"

using namespace vpux;

namespace {

int SIZEOF_SERIALIZED_STRUCT = 192;
int OFFSET_SRC_SERIALIZED_STRUCT = 16;
int OFFSET_DST_SERIALIZED_STRUCT = 24;

#define NUM_MAX_SYMBOLS 2000
#define NUM_MAX_NNDMAOps 1000

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

    unsigned int numNNDMAOps = 0;
    mlir::Value NNDMAOpInput[NUM_MAX_NNDMAOps];
    mlir::Value NNDMAOpOutput[NUM_MAX_NNDMAOps];
};

// createSection() creates an ELF::CreateSectionOp and puts into its body
//   for each object of type DerivedOpType from the FuncOp func an
//   ELFPutOpInSectionOp instruction.
//  In case isNNDMAOp is true, then we store in the NNDMAOpInput and NNDMAOpOutput
//    the input, respectively the output of the current NNDMAOp operation.
template <typename DerivedOpType>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSection(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                            std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                            vpux::ELF::SectionFlagsAttr secFlags, bool isNNDMAOp) {
    mlir::Block& blkFunc = (func.getCallableRegion())->front();

    mlir::OpBuilder builderFunc(&(blkFunc.back()));

    mlir::Operation* endOp = blkFunc.getTerminator();
    llvm::StringRef secName = secNameStr;
    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    ELF::CreateSectionOp ELFCreateSectionOp =
            builderFunc.create<ELF::CreateSectionOp>(endOp->getLoc(),
                                                     sectionType,  // mlir::Type
                                                     secName,      // llvm::StringRef secName,
                                                     secType,      // vpux::ELF::SectionTypeAttr secType,
                                                     secFlags,     // vpux::ELF::SectionFlagsAttr secFlags,
                                                     1,            // int64_t secInfo,
                                                     64            // int64_t secAddrAlign
            );

    mlir::Region& aReg = ELFCreateSectionOp.getOperation()->getRegion(0);

    mlir::Block* blkNew = new mlir::Block();

    aReg.push_back(blkNew);

    for (mlir::Operation& op : blkFunc.getOperations()) {
        DerivedOpType opCasted = llvm::dyn_cast<DerivedOpType>(op);

        if (opCasted) {
            if (isNNDMAOp) {
                // isNNDMAOp is true only when DerivedOpType is vpux::VPUIPRegMapped::NNDMAOp
                NNDMAOpInput[numNNDMAOps] = ((vpux::VPUIPRegMapped::NNDMAOp)opCasted).input();
                NNDMAOpOutput[numNNDMAOps] = ((vpux::VPUIPRegMapped::NNDMAOp)opCasted).output_buff();
                _log.fatal("createSection(): NNDMAOpInput[numNNDMAOps] = {0}\n", NNDMAOpInput[numNNDMAOps]);
                _log.fatal("createSection(): NNDMAOpOutput[numNNDMAOps] = {0}\n", NNDMAOpOutput[numNNDMAOps]);

                numNNDMAOps++;
            }

            mlir::OpBuilder builderELFSectionOpReg(blkNew, blkNew->begin());

            _log.fatal("createSection(): Before builderELFSectionOpReg.create()\n");

            ELF::PutOpInSectionOp ELFPutOpInSectionOp = builderELFSectionOpReg.create<ELF::PutOpInSectionOp>(
                    builderELFSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
                    opCasted.getOperation()->getResult(0)    // mlir::Value inputArg
            );

            VPUX_UNUSED(ELFPutOpInSectionOp);
        }
    }

    return ELFCreateSectionOp.getOperation()->getResult(0);
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

        _log.fatal("    blockArgNum = {0}\n", blockArgNum);

        // By convention the arguments of the FuncOp are described in-order in
        //   the IE.CNNNetwork block, within its parameters inputsInfo and
        //   outputsInfo (with DataInfo operations).
        vpux::IE::DataInfoOp respectiveNetArg;
        if (blockArgNum < diOpInVec.size()) {
            respectiveNetArg = diOpInVec[blockArgNum];
        } else {
            respectiveNetArg = diOpOutVec[blockArgNum - diOpInVec.size()];
        }

        _log.fatal("    respectiveNetArg = {0}\n", respectiveNetArg);

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
    _log.fatal("createRelocationSection(): funcOp = {0}\n", funcOp);

    mlir::Block& blkFunc = (funcOp.getCallableRegion())->front();

    mlir::OpBuilder builderFunc(&(blkFunc.back()));

    mlir::Operation* endOp = blkFunc.getTerminator();

    // Creating the required ELF.SymbolOps
    ELF::SymbolOp ELFSymbolOp[NUM_MAX_SYMBOLS];
    unsigned int numELFSymbolOps = 0;

    unsigned int idx;
    for (idx = 0; idx < numNNDMAOps; idx++) {
        std::string nameSymStr = "nndmaOp";
        nameSymStr += std::to_string(idx);

        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        mlir::StringAttr nameSym;

        nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");

        vpux::ELF::SymbolTypeAttrAttr typeSym;  // TODO: initialize (we currently don't because it's optional)
        mlir::IntegerAttr sizeSym;              // TODO: initialize (we currently don't because it's optional)
        mlir::IntegerAttr valueSym;             // TODO: initialize (we currently don't because it's optional)

        std::string tmpStr;

        _log.fatal("NNDMAOpInput[idx] = {0}\n", NNDMAOpInput[idx]);

        // We handle the NNDMAOpInput[idx] value
        bool res = checkIfValueIsNetArg(NNDMAOpInput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");
        }
        _log.fatal("nameSym = {0}\n", nameSym);

        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                         symbolType,         // mlir::Type
                                                                         NNDMAOpInput[idx],  // mlir::Value inputArg
                                                                         nameSym,            // mlir::StringAttr
                                                                         typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                                                         sizeSym,  // size
                                                                         valueSym  // value
        );
        _log.fatal("ELFSymbolOp[{0}] = {1}\n", numELFSymbolOps, ELFSymbolOp[numELFSymbolOps]);
        numELFSymbolOps++;

        _log.fatal("NNDMAOpOutput[idx] = {0}\n", NNDMAOpOutput[idx]);

        // We handle the NNDMAOpInput[idx] value
        res = checkIfValueIsNetArg(NNDMAOpOutput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_output");
        }
        _log.fatal("nameSym = {0}\n", nameSym);

        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                         symbolType,          // mlir::Type
                                                                         NNDMAOpOutput[idx],  // mlir::Value inputArg
                                                                         nameSym,             // mlir::StringAttr
                                                                         typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                                                         sizeSym,  // size
                                                                         valueSym  // value
        );
        _log.fatal("ELFSymbolOp[{0}] = {1}\n", numELFSymbolOps, ELFSymbolOp[numELFSymbolOps]);
        numELFSymbolOps++;
    }

    // Now, we create the symbol tables.
    // First we create the symbol table for the "rest"/normal symbols.
    vpux::ELF::SectionType sectionType2 = vpux::ELF::SectionType::get(ctx);

    ELF::CreateSymbolTableSectionOp createRestSymTableSectionOp =
            builderFunc.create<ELF::CreateSymbolTableSectionOp>(endOp->getLoc(),
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
            endOp->getLoc(),
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
            endOp->getLoc(),
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
    for (idx = 0; idx < numELFSymbolOps; idx++) {
        ELF::PutOpInSectionOp putOpInSectionOp;

        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        _log.fatal("inputArgELFSymbolOp = {0}\n", inputArgELFSymbolOp);

        if (blockArg) {
            unsigned int blockArgNum = blockArg.getArgNumber();

            _log.fatal("    inputArgELFSymbolOp is a BlockArgument and blockArgNum = {0}\n", blockArgNum);

            if (blockArgNum < diOpInVec.size()) {
                putOpInSectionOp = builderInputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderInputSymTabSec.getUnknownLoc(),         // endOp->getLoc(), // 2022_02_09
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            } else {
                putOpInSectionOp = builderOutputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderOutputSymTabSec.getUnknownLoc(),        // endOp->getLoc(), // 2022_02_09
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            }
        } else {
            putOpInSectionOp = builderRestSymTabSec.create<ELF::PutOpInSectionOp>(
                    builderRestSymTabSec.getUnknownLoc(),          // endOp->getLoc(),
                    ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
            );
        }

        _log.fatal("createRelocationSection(): putOpInSectionOp = {0}\n", putOpInSectionOp);
    }

    // vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    // We now create the relocation sections
    // First, we create the "rest" relocation section
    ELF::CreateRelocationSectionOp createRestRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType2,                                              // mlir::Type
            ".rlt.dma",                                                // secName, // llvm::StringRef
            createRestSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                       // targetSection,
            // vpux::ELF::SectionFlagsAttr::SHF_NONE                      // vpux::ELF::SectionFlagsAttr secFlags, //
            // 2022_02_09
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK);
    //
    _log.fatal("createRelocationSection(): createRestRelocationSectionOp = {0}\n", createRestRelocationSectionOp);
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
            endOp->getLoc(),
            sectionType2,                                               // mlir::Type
            ".rlt.input",                                               // llvm::StringRef secName,
            createInputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    _log.fatal("createRelocationSection(): createInputRelocationSectionOp = {0}\n", createInputRelocationSectionOp);
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
            endOp->getLoc(),
            sectionType2,                                                // mlir::Type
            ".rlt.output",                                               // llvm::StringRef secName,
            createOutputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                         // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELF::SectionFlagsAttr::VPU_SHF_JIT |
                    vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    _log.fatal("createRelocationSection(): createOutputRelocationSectionOp = {0}\n", createOutputRelocationSectionOp);
    //
    mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regOutputRelocSec.push_back(blkOutputRelocSec);
    //
    mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

    // We now create the RelocOps
    for (idx = 0; idx < numELFSymbolOps; idx++) {
        bool isBlkFuncArg = false;

        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        _log.fatal("inputArgELFSymbolOp = {0}\n", inputArgELFSymbolOp);
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        unsigned int blockArgNum;

        if (blockArg) {
            isBlkFuncArg = true;
            blockArgNum = blockArg.getArgNumber();
            _log.fatal("    blockArgNum = {0}\n", blockArgNum);
        }

        ELF::RelocOp ELFRelocOp;

        if (isBlkFuncArg) {
            if (blockArgNum < diOpInVec.size()) {
                ELFRelocOp = builderInputRelocSec.create<ELF::RelocOp>(
                        builderInputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                                (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),  // offsetTargetField // MEGA-TODO
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,         // relocationType
                        ELFSymbolOp[idx].getOperation()->getResult(0),   // ::mlir::Value sourceSymbol
                        0                                                // int64_t addend
                );
            } else {
                ELFRelocOp = builderOutputRelocSec.create<ELF::RelocOp>(
                        builderOutputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                                (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),  // offsetTargetField // MEGA-TODO
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,         // relocationType
                        ELFSymbolOp[idx].getOperation()->getResult(0),   // ::mlir::Value sourceSymbol
                        0                                                // int64_t addend
                );
            }
        } else {
            ELFRelocOp = builderRestRelocSec.create<ELF::RelocOp>(
                    builderRestRelocSec.getUnknownLoc(),
                    ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                            (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),  // offsetTargetField // MEGA-TODO
                    vpux::ELF::RelocationTypeAttr::R_VPU_64,         // relocationType
                    ELFSymbolOp[idx].getOperation()->getResult(0),   // ::mlir::Value sourceSymbol
                    0                                                // int64_t addend
            );
        }

        _log.fatal("createRelocationSection(): ELFRelocOp = {0}\n", ELFRelocOp);
    }

    _log.fatal("createRelocationSection(): funcOp = {0}\n", funcOp);
}

void Convert2VPUIPRegMappedAndELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    // llvm::dbgs() << "Entered Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc().\n";

    vpux::VPUIPRegMapped::NNDMAOp nndmaOpTmp;
    SIZEOF_SERIALIZED_STRUCT = (int)nndmaOpTmp.getBinarySize();

    host_parsing::DmaWrapper tmp;
    OFFSET_SRC_SERIALIZED_STRUCT = (int)((char*)(&tmp.transaction.src) - (char*)(&tmp));
    OFFSET_DST_SERIALIZED_STRUCT = (int)((char*)(&tmp.transaction.dst) - (char*)(&tmp));

    _log.fatal("SIZEOF_SERIALIZED_STRUCT = {0}\n", SIZEOF_SERIALIZED_STRUCT);
    _log.fatal("OFFSET_SRC_SERIALIZED_STRUCT = {0}\n", OFFSET_SRC_SERIALIZED_STRUCT);
    _log.fatal("OFFSET_DST_SERIALIZED_STRUCT = {0}\n", OFFSET_DST_SERIALIZED_STRUCT);

    _log.fatal("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = {0}\n", moduleOp);

    for (mlir::Operation& op : moduleOp) {
        if (vpux::IE::CNNNetworkOp cnnOp = llvm::dyn_cast<vpux::IE::CNNNetworkOp>(op)) {
            _log.fatal("Found a IE::CNNNetworkOp operation\n");

            diOpInVec = cnnOp.getInputsInfo();
            diOpOutVec = cnnOp.getOutputsInfo();

            _log.fatal("  diOpInVec.size() = {0}\n", diOpInVec.size());
            _log.fatal("  diOpOutVec.size() = {0}\n", diOpOutVec.size());
        } else if (mlir::isa<mlir::FuncOp>(op)) {
            funcOp = llvm::cast<mlir::FuncOp>(op);

            // We build 4 different createSections, one for VPUIPRegMapped::DeclareBufferOp,
            //   one for Const::DeclareOp, one for VPUIPRegMapped::NNDMAOp and
            //   one for VPUIPRegMapped::ConfigureBarrierOp.
            createSection<vpux::VPUIPRegMapped::DeclareBufferOp>(funcOp, ctx, ".data.Weights",
                                                                 vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                                 vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
            createSection<vpux::Const::DeclareOp>(funcOp, ctx, ".data.Weights_ct",
                                                  vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                  vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
            mlir::Value nndmaSectionOpValue = createSection<vpux::VPUIPRegMapped::NNDMAOp>(
                    funcOp, ctx, ".text.dmaTasks", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_ALLOC | vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR, true);
            createSection<vpux::VPUIPRegMapped::ConfigureBarrierOp>(funcOp, ctx, ".text.BarrierConfigs",
                                                                    vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                                    vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

            // Now, for each NNDMAOp input and output we want to perform relocation
            createRelocationSection(funcOp, ctx, nndmaSectionOpValue);
        }
    }

    _log.fatal("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = {0}\n", moduleOp);
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
