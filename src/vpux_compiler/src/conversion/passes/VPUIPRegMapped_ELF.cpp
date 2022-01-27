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

#define NUM_MAX_SYMBOLS 2000

int SIZEOF_SERIALIZED_STRUCT = 192;
int OFFSET_SRC_SERIALIZED_STRUCT = 16;
int OFFSET_DST_SERIALIZED_STRUCT = 24;

#define NUM_MAX_NNDMAOps 1000

//
// Convert2VPUIPRegMappedAndELFPass
//

class Convert2VPUIPRegMappedAndELFPass final :
        public Convert2VPUIPRegMappedAndELFBase<Convert2VPUIPRegMappedAndELFPass> {
public:
    explicit Convert2VPUIPRegMappedAndELFPass(Logger log) {
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

    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

    unsigned int numNNDMAOps = 0;
    mlir::Value NNDMAOpInput[NUM_MAX_NNDMAOps];
    mlir::Value NNDMAOpOutput[NUM_MAX_NNDMAOps];
};

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
                // llvm::dbgs() << "createSection(): NNDMAOpInput[numNNDMAOps] = "
                //              << NNDMAOpInput[numNNDMAOps] << "\n";
                // llvm::dbgs() << "createSection(): NNDMAOpOutput[numNNDMAOps] = "
                //              << NNDMAOpOutput[numNNDMAOps] << "\n";
                // llvm::dbgs().flush();

                numNNDMAOps++;
            }

            mlir::OpBuilder builder2(blkNew, blkNew->begin());

            // llvm::dbgs() << "createSection(): Before builder2.create()\n";
            // llvm::dbgs().flush();

            ELF::PutOpInSectionOp ELFPutOpInSectionOp = builder2.create<ELF::PutOpInSectionOp>(
                    builder2.getUnknownLoc(),              // endOp->getLoc(),
                    opCasted.getOperation()->getResult(0)  // mlir::Value inputArg
            );

            VPUX_UNUSED(ELFPutOpInSectionOp);
        }
    }

    return ELFCreateSectionOp.getOperation()->getResult(0);
}

bool Convert2VPUIPRegMappedAndELFPass::checkIfValueIsNetArg(mlir::Value val, std::string& strRes) {
    bool res;

    mlir::BlockArgument blockArg = val.dyn_cast_or_null<mlir::BlockArgument>();
    if (blockArg) {
        res = true;

        unsigned int blockArgNum = blockArg.getArgNumber();

        // llvm::dbgs() << "    blockArgNum = " << blockArgNum << "\n";

        vpux::IE::DataInfoOp respectiveNetArg;
        if (blockArgNum < diOpInVec.size()) {
            respectiveNetArg = diOpInVec[blockArgNum];
        } else {
            respectiveNetArg = diOpOutVec[blockArgNum - diOpInVec.size()];
        }

        // llvm::dbgs() << "    respectiveNetArg = " << respectiveNetArg << "\n";

        strRes = respectiveNetArg.name().str();
    } else {
        res = false;
    }

    return res;
}

void Convert2VPUIPRegMappedAndELFPass::createRelocationSection(mlir::FuncOp funcOp, mlir::MLIRContext* ctx,
                                                               mlir::Value& nndmaSectionOpValue) {
    // llvm::dbgs() << "createRelocationSection(): funcOp = " << funcOp << "\n";

    mlir::Block& blkFunc = (funcOp.getCallableRegion())->front();

    mlir::OpBuilder builderFunc(&(blkFunc.back()));

    mlir::Operation* endOp = blkFunc.getTerminator();

    ELF::SymbolOp ELFSymbolOp[NUM_MAX_SYMBOLS];

    unsigned int numELFSymbolOps = 0;

    unsigned int idx;
    for (idx = 0; idx < numNNDMAOps; idx++) {
        // llvm::dbgs() << "NNDMAOpInput[idx] = " << NNDMAOpInput[idx] << "\n";

        std::string nameSymStr = "nndmaOp";
        nameSymStr += std::to_string(idx);

        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);
        mlir::StringAttr nameSym;

        nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");

        vpux::ELF::SymbolTypeAttrAttr typeSym;  // TODO: initialize (we currently don't because it's optional)
        mlir::IntegerAttr sizeSym;              // TODO: initialize (we currently don't because it's optional)
        mlir::IntegerAttr valueSym;             // TODO: initialize (we currently don't because it's optional)

        std::string tmpStr;
        bool res = checkIfValueIsNetArg(NNDMAOpInput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_input");
        }

        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                         symbolType,         // mlir::Type
                                                                         NNDMAOpInput[idx],  // mlir::Value inputArg
                                                                         nameSym,            // mlir::StringAttr
                                                                         typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                                                         sizeSym,  // size
                                                                         valueSym  // value
        );
        numELFSymbolOps++;

        // llvm::dbgs() << "NNDMAOpOutput[idx] = " << NNDMAOpOutput[idx] << "\n";

        res = checkIfValueIsNetArg(NNDMAOpOutput[idx], tmpStr);
        if (res) {
            nameSym = mlir::StringAttr::get(ctx, tmpStr);
        } else {
            nameSym = mlir::StringAttr::get(ctx, nameSymStr + "_output");
        }

        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                         symbolType,          // mlir::Type
                                                                         NNDMAOpOutput[idx],  // mlir::Value inputArg
                                                                         nameSym,             // mlir::StringAttr
                                                                         typeSym,  // vpux::ELF::SymbolTypeAttrAttr
                                                                         sizeSym,  // size
                                                                         valueSym  // value
        );
        numELFSymbolOps++;
    }

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
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regRestSymTabSec.push_back(blkRestSymTabSec);
    //
    mlir::OpBuilder builderRestSymTabSec(blkRestSymTabSec, blkRestSymTabSec->begin());

    ELF::CreateSymbolTableSectionOp createInputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            endOp->getLoc(),
            sectionType2,                                   // mlir::Type
            ".input.symbolTableSection",                    // llvm::StringRef secName,
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

    ELF::CreateSymbolTableSectionOp createOutputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            endOp->getLoc(),
            sectionType2,                                    // mlir::Type
            ".output.symbolTableSection",                    // llvm::StringRef secName,
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

    for (idx = 0; idx < numELFSymbolOps; idx++) {
        ELF::PutOpInSectionOp putOpInSectionOp;

        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        // llvm::dbgs() << "inputArgELFSymbolOp = " << inputArgELFSymbolOp << "\n";

        if (blockArg) {
            unsigned int blockArgNum = blockArg.getArgNumber();

            // llvm::dbgs() << "    inputArgELFSymbolOp is a BlockArgument and blockArgNum = " << blockArgNum << "\n";

            if (blockArgNum < diOpInVec.size()) {
                putOpInSectionOp = builderInputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderRestSymTabSec.getUnknownLoc(),          // endOp->getLoc(),
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            } else {
                putOpInSectionOp = builderOutputSymTabSec.create<ELF::PutOpInSectionOp>(
                        builderRestSymTabSec.getUnknownLoc(),          // endOp->getLoc(),
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            }
        } else {
            putOpInSectionOp = builderRestSymTabSec.create<ELF::PutOpInSectionOp>(
                    builderRestSymTabSec.getUnknownLoc(),          // endOp->getLoc(),
                    ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
            );
        }

        // llvm::dbgs() << "createRelocationSection(): putOpInSectionOp = " << putOpInSectionOp << "\n";
        // llvm::dbgs().flush();
    }

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    ELF::CreateRelocationSectionOp createRestRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                               // mlir::Type
            ".rela.dma",                                               // secName, // llvm::StringRef secName,
            createRestSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                       // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_NONE                      // vpux::ELF::SectionFlagsAttr secFlags,
    );

    // llvm::dbgs() << "createRelocationSection(): createRestRelocationSectionOp = " << createRestRelocationSectionOp
    //              << "\n";
    // llvm::dbgs().flush();
    //
    mlir::Region& regRestRelocSec = createRestRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkRestRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regRestRelocSec.push_back(blkRestRelocSec);
    //
    mlir::OpBuilder builderRestRelocSec(blkRestRelocSec, blkRestRelocSec->begin());

    ELF::CreateRelocationSectionOp createInputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                                // mlir::Type
            ".rela.input",                                              // llvm::StringRef secName,
            createInputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USERINPUT              // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    // llvm::dbgs() << "createRelocationSection(): createInputRelocationSectionOp = " << createInputRelocationSectionOp
    //              << "\n";
    // llvm::dbgs().flush();
    //
    mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regInputRelocSec.push_back(blkInputRelocSec);
    //
    mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

    ELF::CreateRelocationSectionOp createOutputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                                 // mlir::Type
            ".rela.output",                                              // llvm::StringRef secName,
            createOutputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                         // targetSection,
            vpux::ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT              // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    // llvm::dbgs() << "createRelocationSection(): createOutputRelocationSectionOp = " <<
    // createOutputRelocationSectionOp
    //             << "\n";
    // llvm::dbgs().flush();
    //
    mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputRelocSec = new mlir::Block();
    //
    // This instruction has to be before defining builder... to avoid SegFault
    regOutputRelocSec.push_back(blkOutputRelocSec);
    //
    mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

    for (idx = 0; idx < numELFSymbolOps; idx++) {
        bool isBlkFuncArg = false;

        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        // llvm::dbgs() << "inputArgELFSymbolOp = " << inputArgELFSymbolOp << "\n";
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        unsigned int blockArgNum;

        if (blockArg) {
            isBlkFuncArg = true;
            blockArgNum = blockArg.getArgNumber();
            // llvm::dbgs() << "    blockArgNum = " << blockArgNum << "\n";
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

        // llvm::dbgs() << "createRelocationSection(): ELFRelocOp = " << ELFRelocOp << "\n";
        // llvm::dbgs().flush();
    }

    // llvm::dbgs() << "createRelocationSection(): funcOp = " << funcOp << "\n";
    // llvm::dbgs().flush();
}

void Convert2VPUIPRegMappedAndELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    // llvm::dbgs() << "Entered Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc().\n";

    vpux::VPUIPRegMapped::NNDMAOp nndmaOpTmp;
    SIZEOF_SERIALIZED_STRUCT = nndmaOpTmp.getBinarySize();

    host_parsing::DmaWrapper tmp;
    OFFSET_SRC_SERIALIZED_STRUCT = (char*)(&tmp.transaction.src) - (char*)(&tmp);
    OFFSET_DST_SERIALIZED_STRUCT = (char*)(&tmp.transaction.dst) - (char*)(&tmp);

    // llvm::dbgs() << "SIZEOF_SERIALIZED_STRUCT = " << SIZEOF_SERIALIZED_STRUCT << "\n";
    // llvm::dbgs() << "OFFSET_SRC_SERIALIZED_STRUCT = " << OFFSET_SRC_SERIALIZED_STRUCT << "\n";
    // llvm::dbgs() << "OFFSET_DST_SERIALIZED_STRUCT = " << OFFSET_DST_SERIALIZED_STRUCT << "\n";
    // llvm::dbgs().flush();

    // llvm::dbgs() << "Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = " << moduleOp << "\n";
    // llvm::dbgs().flush();

    for (mlir::Operation& op : moduleOp) {
        if (vpux::IE::CNNNetworkOp cnnOp = llvm::dyn_cast<vpux::IE::CNNNetworkOp>(op)) {
            // llvm::dbgs() << "Found a IE::CNNNetworkOp operation\n";
            // llvm::dbgs().flush();

            diOpInVec = cnnOp.getInputsInfo();
            diOpOutVec = cnnOp.getOutputsInfo();

            // llvm::dbgs() << "  diOpInVec.size() = " << diOpInVec.size() << "\n";
            // llvm::dbgs() << "  diOpOutVec.size() = " << diOpOutVec.size() << "\n";
            // llvm::dbgs().flush();
        } else if (mlir::isa<mlir::FuncOp>(op)) {
            funcOp = llvm::cast<mlir::FuncOp>(op);

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

    // llvm::dbgs() << "Exiting Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): moduleOp = " << moduleOp << "\n";
    // llvm::dbgs().flush();
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
