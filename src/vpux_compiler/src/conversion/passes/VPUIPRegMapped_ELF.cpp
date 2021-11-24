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
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"  // 2021_11_08
#include "vpux/compiler/dialect/const/ops.hpp"

#include "llvm/Support/Debug.h"  // Alex: 2021_11_08

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
//#include "mlir/IR/PatternMatch.h"

// clang-format off

using namespace vpux;

namespace {


#define NUM_MAX_SYMBOLS 2000

#define SIZEOF_SERIALIZED_STRUCT 80
#define OFFSET_SRC_SERIALIZED_STRUCT 16
#define OFFSET_DST_SERIALIZED_STRUCT 24


#define NUM_MAX_NNDMAOps 1000

//
// Convert2VPUIPRegMappedAndELFPass
//

class Convert2VPUIPRegMappedAndELFPass final : public Convert2VPUIPRegMappedAndELFBase<Convert2VPUIPRegMappedAndELFPass> {
public:
    explicit Convert2VPUIPRegMappedAndELFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename DerivedOpType>
    mlir::Value createSectionOp(mlir::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr, vpux::ELF::SectionTypeAttr secType, vpux::ELF::SectionFlagsAttr secFlags, bool isNNDMAOp = false);

    // void createRelocationSection(mlir::FuncOp func, mlir::MLIRContext *ctx, std::string secNameStr, vpux::ELF::SectionFlagsAttr secFlags, mlir::Value& nndmaSectionOp);
    void createRelocationSection(mlir::FuncOp func, mlir::MLIRContext* ctx, mlir::Value& nndmaSectionOp);

    //void safeRunOnFunc() final;
    void safeRunOnModule() final;

    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpInVec;
    vpux::SmallVector<vpux::IE::DataInfoOp, 1> diOpOutVec;

    unsigned int numNNDMAOps = 0;
    mlir::Value NNDMAOpInput[NUM_MAX_NNDMAOps];
    mlir::Value NNDMAOpOutput[NUM_MAX_NNDMAOps];
};



// The template is inspired from https://stackoverflow.com/questions/22127041/c-pass-variable-type-to-function/22127100
template <typename DerivedOpType>
mlir::Value Convert2VPUIPRegMappedAndELFPass::createSectionOp(mlir::FuncOp func, mlir::MLIRContext* ctx,
                                                              std::string secNameStr, vpux::ELF::SectionTypeAttr secType,
                                                              vpux::ELF::SectionFlagsAttr secFlags, bool isNNDMAOp) {
    llvm::dbgs() << "createSectionOp(): func = " << func << "\n";

    // printRegion(*(func.getCallableRegion()));

    mlir::Block& blkFunc = (func.getCallableRegion())->front();

    /*
    // Inspired from kmb-plugin/thirdparty/llvm-project/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(&blkFunc);
    */
    // See https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    //  Inspired from https://mlir.llvm.org/doxygen/classmlir_1_1Block.html
    mlir::OpBuilder builderFunc(&(blkFunc.back()));

    mlir::Operation* endOp = blkFunc.getTerminator();
    llvm::StringRef secName = secNameStr;
    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    // void CreateSectionOp::build(::mlir::OpBuilder &odsBuilder,
    //   ::mlir::OperationState &odsState, ::mlir::Type section,
    //   ::llvm::StringRef secName, vpux::ELF::SectionTypeAttr secType,
    //   vpux::ELF::SectionFlagsAttr secFlags, int64_t secInfo,
    //   int64_t secAddrAlign)
    ELF::CreateSectionOp ELFCreateSectionOp = builderFunc.create<ELF::CreateSectionOp>(
            // origOp->getLoc(),
            endOp->getLoc(),
            sectionType,  // mlir::Type
            secName,      // llvm::StringRef secName,
            secType,      // vpux::ELF::SectionTypeAttr secType,
            secFlags,     // vpux::ELF::SectionFlagsAttr secFlags,
            1,            // int64_t secInfo,
            64            // int64_t secAddrAlign
    );
    llvm::dbgs() << "CreateSectionOp(): ELFCreateSectionOp = " << ELFCreateSectionOp << "\n";
    llvm::dbgs().flush();

    mlir::Region& aReg = ELFCreateSectionOp.getOperation()->getRegion(0);

    mlir::Block* blkNew = new mlir::Block();

    // Alex: This instruction has to be before defining builder2 to avoid SegFault
    aReg.push_back(blkNew);

    // blkFunc.walk([&](DerivedOpType opCasted)
    for (mlir::Operation& op : blkFunc.getOperations()) {
        // vpux::VPUIPRegMapped::DeclareBufferOp opDTO =
        //        llvm::dyn_cast<vpux::VPUIPRegMapped::DeclareBufferOp>(op);

        DerivedOpType opCasted = llvm::dyn_cast<DerivedOpType>(op);

        if (opCasted) {
            if (isNNDMAOp) {
                // isNNDMAOp is true only when DerivedOpType is vpux::VPUIPRegMapped::NNDMAOp)opCasted
                NNDMAOpInput[numNNDMAOps] = ((vpux::VPUIPRegMapped::NNDMAOp)opCasted).input();
                NNDMAOpOutput[numNNDMAOps] = ((vpux::VPUIPRegMapped::NNDMAOp)opCasted).output_buff();
                llvm::dbgs() << "createSectionOp(): NNDMAOpInput[numNNDMAOps] = " << NNDMAOpInput[numNNDMAOps] << "\n";
                llvm::dbgs() << "createSectionOp(): NNDMAOpOutput[numNNDMAOps] = " << NNDMAOpOutput[numNNDMAOps]
                             << "\n";
                llvm::dbgs().flush();

                numNNDMAOps++;
            }

            // llvm::dbgs() << "createSectionOp(): op = " << op << " is a DeclareBufferOp\n";
            // llvm::dbgs() << "safeRunOnFunc(): Before builder2(blkNew, blkNew->begin())\n";
            // llvm::dbgs().flush();

            // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
            mlir::OpBuilder builder2(blkNew, blkNew->begin());

            llvm::dbgs() << "createSectionOp(): Before builder2.create()\n";
            llvm::dbgs().flush();

            ELF::PutAnyOpInSectionOp ELFPutAnyOpInSectionOp = builder2.create<ELF::PutAnyOpInSectionOp>(
                    // origOp->getLoc(),
                    builder2.getUnknownLoc(),              // endOp->getLoc(),
                    opCasted.getOperation()->getResult(0)  // mlir::Value inputArg
                    // op.getResult(0) // mlir::Value inputArg
            );

            llvm::dbgs() << "createSectionOp(): ELFPutAnyOpInSectionOp = " << ELFPutAnyOpInSectionOp << "\n";
            llvm::dbgs().flush();

            // See https://mlir.llvm.org/doxygen/OpDefinition_8h_source.html
            /*
             // Gives error <<Assertion `!op->getBlock() && "already in an operation block!"' failed.>>,
             //   which is only normal because the builder.create() assigns the new
             //   instruction to a certain block:
             //   blkNew->push_back(anELFPutAnyOpInSectionOp.getOperation());
            */

            // aReg.push_back(blkNew);

            // Does NOT work: llvm::dbgs() << "safeRunOnFunc(): blkNew = " << *blkNew << "\n";
            // Does NOT work: llvm::dbgs() << "safeRunOnFunc(): aReg = " << aReg << "\n";
            llvm::dbgs().flush();

            llvm::dbgs() << "createSectionOp(): ELFCreateSectionOp = " << ELFCreateSectionOp << "\n";
            llvm::dbgs() << "createSectionOp(): func = " << func << "\n";
            llvm::dbgs().flush();
            //(void)createELFCreateSectionOp;
        }
    }

    // <<error: taking address of rvalue>>: return &(ELFCreateSectionOp.getOperation()->getResult(0));
    return ELFCreateSectionOp.getOperation()->getResult(0);
}

void Convert2VPUIPRegMappedAndELFPass::createRelocationSection(mlir::FuncOp funcOp,
                                                               mlir::MLIRContext* ctx,
                                                               // std::string secNameStr,
                                                               // vpux::ELF::SectionFlagsAttr secFlags,
                                                               mlir::Value& nndmaSectionOpValue) {
    llvm::dbgs() << "createRelocationSection(): funcOp = " << funcOp << "\n";

    // printRegion(*(funcOp.getCallableRegion()));

    mlir::Block& blkFunc = (funcOp.getCallableRegion())->front();

    /*
    // Inspired from kmb-plugin/thirdparty/llvm-project/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(&blkFunc);
    */
    // See https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    //  Inspired from https://mlir.llvm.org/doxygen/classmlir_1_1Block.html
    mlir::OpBuilder builderFunc(&(blkFunc.back()));

    // From https://mlir.llvm.org/doxygen/classmlir_1_1Block.html
    mlir::Operation* endOp = blkFunc.getTerminator();

    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    //   ::mlir::Type section, ::llvm::StringRef secName);

    ELF::SymbolOp ELFSymbolOp[NUM_MAX_SYMBOLS];

    unsigned int numELFSymbolOps = 0;

    unsigned int idx;
    for (idx = 0; idx < numNNDMAOps; idx++) {
        llvm::dbgs() << "NNDMAOpInput[idx] = " << NNDMAOpInput[idx] << "\n";

        vpux::ELF::SymbolType symbolType = vpux::ELF::SymbolType::get(ctx);

        // void SymbolOp::build(::mlir::OpBuilder &odsBuilder,
        //   ::mlir::OperationState &odsState,
        //   ::mlir::Type symbol, ::mlir::Value inputArg)
        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                           symbolType,  // mlir::Type
                                                                           NNDMAOpInput[idx]);
        numELFSymbolOps++;

        llvm::dbgs() << "NNDMAOpOutput[idx] = " << NNDMAOpOutput[idx] << "\n";
        //
        ELFSymbolOp[numELFSymbolOps] = builderFunc.create<ELF::SymbolOp>(endOp->getLoc(),
                                                                           symbolType,  // mlir::Type
                                                                           NNDMAOpOutput[idx]);
        numELFSymbolOps++;
    }

    vpux::ELF::SectionType sectionType2 = vpux::ELF::SectionType::get(ctx);

    // static void build(::mlir::OpBuilder &odsBuilder,
    //   ::mlir::OperationState &odsState, ::mlir::Type section,
    //   ::llvm::StringRef secName, vpux::ELF::SectionFlagsAttr secFlags);
    ELF::CreateSymbolTableSectionOp createRestSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            endOp->getLoc(),
            sectionType2,  // mlir::Type
            ".rest.symbolTableSection", // llvm::StringRef secName,
            //".rela.symbolTableSection"  // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_NONE
    );
    //
    mlir::Region& regRestSymTabSec = createRestSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkRestSymTabSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builderSymTabSec to avoid SegFault
    regRestSymTabSec.push_back(blkRestSymTabSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderRestSymTabSec(blkRestSymTabSec, blkRestSymTabSec->begin());


    // static void build(::mlir::OpBuilder &odsBuilder,
    //   ::mlir::OperationState &odsState, ::mlir::Type section,
    //   ::llvm::StringRef secName, vpux::ELF::SectionFlagsAttr secFlags);
    ELF::CreateSymbolTableSectionOp createInputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            endOp->getLoc(),
            sectionType2,  // mlir::Type
            ".input.symbolTableSection", // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_USERINPUT                  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    mlir::Region& regInputSymTabSec = createInputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputSymTabSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builderSymTabSec to avoid SegFault
    regInputSymTabSec.push_back(blkInputSymTabSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderInputSymTabSec(blkInputSymTabSec, blkInputSymTabSec->begin());


    // static void build(::mlir::OpBuilder &odsBuilder,
    //   ::mlir::OperationState &odsState, ::mlir::Type section,
    //   ::llvm::StringRef secName, vpux::ELF::SectionFlagsAttr secFlags);
    ELF::CreateSymbolTableSectionOp createOutputSymTableSectionOp = builderFunc.create<ELF::CreateSymbolTableSectionOp>(
            endOp->getLoc(),
            sectionType2,  // mlir::Type
            ".output.symbolTableSection",                 // llvm::StringRef secName,
            vpux::ELF::SectionFlagsAttr::SHF_USEROUTPUT   // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    mlir::Region& regOutputSymTabSec = createOutputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputSymTabSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builderSymTabSec to avoid SegFault
    regOutputSymTabSec.push_back(blkOutputSymTabSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderOutputSymTabSec(blkOutputSymTabSec, blkOutputSymTabSec->begin());

    /*
    // See https://mlir.llvm.org/doxygen/FunctionSupport_8h_source.html
    unsigned int numArgs = funcOp.getNumArguments();
    llvm::dbgs() << "numArgs = " << numArgs << "\n";
    //
    for (unsigned int idx = 0; idx < numArgs; idx++) {
        llvm::dbgs() << "funcOp.getArgument(" << idx << ") = " << funcOp.getArgument(idx) << "\n";
    }
    */

    for (idx = 0; idx < numELFSymbolOps; idx++) {
        ELF::PutAnyOpInSectionOp putAnyOpInSectionOp;

        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        llvm::dbgs() << "inputArgELFSymbolOp = "
                     << inputArgELFSymbolOp << "\n";
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        if (blockArg) {
            unsigned int blockArgNum = blockArg.getArgNumber();

            llvm::dbgs() << "    inputArgELFSymbolOp is a BlockArgument and blockArgNum = "
                         << blockArgNum << "\n";

            if (blockArgNum < diOpInVec.size()) {
                // void PutAnyOpInSectionOp::build(::mlir::OpBuilder &odsBuilder,
                //     ::mlir::OperationState &odsState, ::mlir::Value inputArg)
                putAnyOpInSectionOp = builderInputSymTabSec.create<ELF::PutAnyOpInSectionOp>(
                        // origOp->getLoc(),
                        builderRestSymTabSec.getUnknownLoc(),                // endOp->getLoc(),
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            }
            else {
                // void PutAnyOpInSectionOp::build(::mlir::OpBuilder &odsBuilder,
                //     ::mlir::OperationState &odsState, ::mlir::Value inputArg)
                putAnyOpInSectionOp = builderOutputSymTabSec.create<ELF::PutAnyOpInSectionOp>(
                        // origOp->getLoc(),
                        builderRestSymTabSec.getUnknownLoc(),                // endOp->getLoc(),
                        ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
                );
            }
        }
        else {
            // void PutAnyOpInSectionOp::build(::mlir::OpBuilder &odsBuilder,
            //     ::mlir::OperationState &odsState, ::mlir::Value inputArg)
            putAnyOpInSectionOp = builderRestSymTabSec.create<ELF::PutAnyOpInSectionOp>(
                    // origOp->getLoc(),
                    builderRestSymTabSec.getUnknownLoc(),                // endOp->getLoc(),
                    ELFSymbolOp[idx].getOperation()->getResult(0)  // mlir::Value inputArg
            );
        }

        // (void)putAnyOpInSectionOp;
        llvm::dbgs() << "createRelocationSection(): putAnyOpInSectionOp = " << putAnyOpInSectionOp << "\n";
        llvm::dbgs().flush();
    }

    // static void CreateRelocationSectionOp::build(::mlir::OpBuilder &odsBuilder,
    //   ::mlir::OperationState &odsState,
    //   ::mlir::Type section, ::llvm::StringRef secName, ::mlir::Value sourceSymbolTableSection,
    //   ::mlir::Value targetSection, vpux::ELF::SectionFlagsAttr secFlags);
    // Note: section is actually the result of the operation.
    // llvm::StringRef secName = secNameStr;

    vpux::ELF::SectionType sectionType = vpux::ELF::SectionType::get(ctx);

    ELF::CreateRelocationSectionOp createRestRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                                // mlir::Type
            ".rela.dma",                                                // secName, // llvm::StringRef secName,
            createRestSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_NONE  // secFlags // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    // (void)createRestRelocationSectionOp;
    llvm::dbgs() << "createRelocationSection(): createRestRelocationSectionOp = " << createRestRelocationSectionOp
                 << "\n";
    llvm::dbgs().flush();
    //
    mlir::Region& regRestRelocSec = createRestRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkRestRelocSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builder... to avoid SegFault
    regRestRelocSec.push_back(blkRestRelocSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderRestRelocSec(blkRestRelocSec, blkRestRelocSec->begin());

    ELF::CreateRelocationSectionOp createInputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                                // mlir::Type
            ".rela.input",                                              // llvm::StringRef secName,
            createInputSymTableSectionOp.getOperation()->getResult(0),  // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_USERINPUT                  // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    // (void)createInputRelocationSectionOp;
    llvm::dbgs() << "createRelocationSection(): createInputRelocationSectionOp = " << createInputRelocationSectionOp
                 << "\n";
    llvm::dbgs().flush();
    //
    mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputRelocSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builder... to avoid SegFault
    regInputRelocSec.push_back(blkInputRelocSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

    ELF::CreateRelocationSectionOp createOutputRelocationSectionOp = builderFunc.create<ELF::CreateRelocationSectionOp>(
            endOp->getLoc(),
            sectionType,                                                // mlir::Type
            ".rela.output",                                             // llvm::StringRef secName,
            createOutputSymTableSectionOp.getOperation()->getResult(0), // sourceSymbolTableSection,
            nndmaSectionOpValue,                                        // targetSection,
            vpux::ELF::SectionFlagsAttr::SHF_USEROUTPUT                 // vpux::ELF::SectionFlagsAttr secFlags,
    );
    //
    // (void)createOutputRelocationSectionOp;
    llvm::dbgs() << "createRelocationSection(): createOutputRelocationSectionOp = " << createOutputRelocationSectionOp
                 << "\n";
    llvm::dbgs().flush();
    //
    mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputRelocSec = new mlir::Block();
    //
    // Alex: This instruction has to be before defining builder... to avoid SegFault
    regOutputRelocSec.push_back(blkOutputRelocSec);
    //
    // From https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

    for (idx = 0; idx < numELFSymbolOps; idx++) {
        bool isBlkFuncArg = false;

        // llvm::dbgs() << "safeRunOnFunc(): ELFSymbolOp[idx].inputArg() = " << ELFSymbolOp[idx].inputArg() << "\n";
        mlir::Value inputArgELFSymbolOp = ELFSymbolOp[idx].inputArg();
        llvm::dbgs() << "inputArgELFSymbolOp = "
                     << inputArgELFSymbolOp << "\n";
        mlir::BlockArgument blockArg = inputArgELFSymbolOp.dyn_cast_or_null<mlir::BlockArgument>();
        unsigned int blockArgNum;

        if (blockArg) {
            isBlkFuncArg = true;
            blockArgNum = blockArg.getArgNumber();
            llvm::dbgs() << "    blockArgNum = "
                         << blockArgNum << "\n";
        }
        /*
        for (unsigned int i = 0; i < numELFSymbolOps; i++) {
            mlir::Value blkFuncArg = blkFunc.getArgument(i);
            if (blkFuncArg == ELFSymbolOp[idx].inputArg()) {
                llvm::dbgs() << "safeRunOnFunc(): ELFSymbolOp[idx].inputArg() is argument of blkFunc\n";
                isBlkFuncArg = true;
            }
        }
        */

        ELF::RelocOp ELFRelocOp;

        if (isBlkFuncArg) {
            if (blockArgNum < diOpInVec.size()) {
                // void RelocOp::build(::mlir::OpBuilder &odsBuilder,
                //   ::mlir::OperationState &odsState, int64_t offsetTargetField,
                //   vpux::ELF::RelocationTypeAttr relocationType,
                //   ::mlir::Value sourceSymbol, int64_t addend)
                ELFRelocOp = builderInputRelocSec.create<ELF::RelocOp>(
                        builderInputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                                (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),   // offsetTargetField // MEGA-TODO
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,          // relocationType
                        ELFSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                        0                                                 // int64_t addend
                );
            }
            else {
                // void RelocOp::build(::mlir::OpBuilder &odsBuilder,
                //   ::mlir::OperationState &odsState, int64_t offsetTargetField,
                //   vpux::ELF::RelocationTypeAttr relocationType,
                //   ::mlir::Value sourceSymbol, int64_t addend)
                ELFRelocOp = builderOutputRelocSec.create<ELF::RelocOp>(
                        builderOutputRelocSec.getUnknownLoc(),
                        ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                                (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),   // offsetTargetField // MEGA-TODO
                        vpux::ELF::RelocationTypeAttr::R_VPU_64,          // relocationType
                        ELFSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                        0                                                 // int64_t addend
                );
            }
        } else {
            // void RelocOp::build(::mlir::OpBuilder &odsBuilder,
            //   ::mlir::OperationState &odsState, int64_t offsetTargetField,
            //   vpux::ELF::RelocationTypeAttr relocationType,
            //   ::mlir::Value sourceSymbol, int64_t addend)
            ELFRelocOp = builderRestRelocSec.create<ELF::RelocOp>(
                    builderRestRelocSec.getUnknownLoc(),
                    ((idx & 1) == 0 ? OFFSET_SRC_SERIALIZED_STRUCT : OFFSET_DST_SERIALIZED_STRUCT) +
                            (SIZEOF_SERIALIZED_STRUCT * (idx / 2)),   // offsetTargetField // MEGA-TODO
                    vpux::ELF::RelocationTypeAttr::R_VPU_64,          // relocationType
                    ELFSymbolOp[idx].getOperation()->getResult(0),  // ::mlir::Value sourceSymbol
                    0                                                 // int64_t addend
            );
        }

        // (void)ELFRelocOp
        llvm::dbgs() << "createRelocationSection(): ELFRelocOp = " << ELFRelocOp << "\n";
        llvm::dbgs().flush();
    }

    llvm::dbgs() << "createRelocationSection(): funcOp = " << funcOp << "\n";
    llvm::dbgs().flush();
}

void Convert2VPUIPRegMappedAndELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::FuncOp funcOp; // = getFunction();
    mlir::ModuleOp moduleOp = getOperation();

    llvm::dbgs() << "Entered Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc().\n";

    //llvm::dbgs() << "safeRunOnFunc(): funcOp = " << funcOp << "\n";
    llvm::dbgs() << "safeRunOnFunc(): moduleOp = " << moduleOp << "\n";
    llvm::dbgs().flush();

    for (mlir::Operation& op : moduleOp) {
        if (vpux::IE::CNNNetworkOp cnnOp = llvm::dyn_cast<vpux::IE::CNNNetworkOp>(op)) {
            llvm::dbgs() << "Found a IE::CNNNetworkOp operation\n";
            llvm::dbgs().flush();

            diOpInVec = cnnOp.getInputsInfo();
            diOpOutVec = cnnOp.getOutputsInfo();

            llvm::dbgs() << "  diOpInVec.size() = " << diOpInVec.size() << "\n";
            llvm::dbgs() << "  diOpOutVec.size() = " << diOpOutVec.size() << "\n";
            llvm::dbgs().flush();
        } else if (mlir::isa<mlir::FuncOp>(op)) {
            funcOp = llvm::cast<mlir::FuncOp>(op);  // use maybe mlir::cast

            createSectionOp<vpux::VPUIPRegMapped::DeclareBufferOp>(funcOp, ctx, ".data.Weights",
                                                                   vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                                   vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
            createSectionOp<vpux::Const::DeclareOp>(funcOp, ctx, ".data.Weights_ct", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                    vpux::ELF::SectionFlagsAttr::SHF_ALLOC);
            mlir::Value nndmaSectionOpValue = createSectionOp<vpux::VPUIPRegMapped::NNDMAOp>(
                    funcOp, ctx, ".text.dmaTasks", vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                    vpux::ELF::SectionFlagsAttr::SHF_ALLOC | vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR, true);
            createSectionOp<vpux::VPUIPRegMapped::ConfigureBarrierOp>(funcOp, ctx, ".text.BarrierConfigs",
                                                                      vpux::ELF::SectionTypeAttr::SHT_PROGBITS,
                                                                      vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);
            // CreateSectionOp<vpux::VPUIPRegMapped::NNDMAOp>(funcOp, ctx, ".text.BarrierConfigs",
            // vpux::ELF::SectionTypeAttr::SHT_PROGBITS, vpux::ELF::SectionFlagsAttr::SHF_EXECINSTR);

            // Now, for each NNDMAOp input and output we want to perform relocation

            createRelocationSection(funcOp, ctx, nndmaSectionOpValue);
        }
    }
}
}  // namespace

//
// createConvert2VPUIPRegMappedAndELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvert2VPUIPRegMappedAndELFPass(Logger log) {
    return std::make_unique<Convert2VPUIPRegMappedAndELFPass>(log);
}
