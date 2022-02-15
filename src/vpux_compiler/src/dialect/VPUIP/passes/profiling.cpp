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

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

using namespace vpux;

namespace {

//
// UPAProfilingPass
//

class UPAProfilingPass final : public VPUIP::UPAProfilingBase<UPAProfilingPass> {
public:
    explicit UPAProfilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// GroupProfilingBuffersPass
//

class GroupProfilingBuffersPass final : public VPUIP::GroupProfilingBuffersBase<GroupProfilingBuffersPass> {
public:
    explicit GroupProfilingBuffersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void UPAProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<VPURT::TaskOp> upaTasks;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() == vpux::VPU::ExecutorKind::SHAVE_UPA) {
            _log.trace("Adding Operation '{0}'", taskOp->getLoc());
            upaTasks.push_back(taskOp);
        }
    });

    if (upaTasks.empty()) {  // No UPA task in the network
        return;
    }

    unsigned elementSize = VPUIP::HW_UPA_PROFILING_SIZE_BYTES / sizeof(uint32_t);
    unsigned outputSize = static_cast<unsigned>(upaTasks.size() * elementSize);
    auto outputResult = mlir::MemRefType::get({outputSize}, getUInt32Type(ctx));

    auto profilingId = static_cast<int64_t>(netOp.getProfilingOutputsCount());
    unsigned upaId = 0;
    for (auto& upaTask : upaTasks) {
        builder.setInsertionPoint(upaTask);
        auto timestampType = mlir::MemRefType::get({elementSize}, getUInt32Type(ctx));
        int offset = upaId * elementSize * sizeof(uint32_t);
        auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::Identifier::get("declareProfilingBuffer", ctx)), timestampType,
                VPURT::BufferSection::ProfilingOutput, profilingId, offset);

        const auto profilingMeta = llvm::formatv("_PROF_{0}", upaId).str();
        const auto loc = appendLoc(upaTask->getLoc(), profilingMeta);
        upaTask->setLoc(loc);
        upaTask.profiling_dataMutable().assign(declareOp);
        upaId++;
    }

    // Declare and create additional output from network
    auto profilngResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "upa");

    // And to the returnOp
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    returnOp.operandsMutable().append(profilngResult);
}

void GroupProfilingBuffersPass::safeRunOnModule() {
    auto ctx = &getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    if (netOp.profilingOutputsInfo().empty())
        return;

    // Collecting sizes of all profiling buffers in order to calculate offsets in the base buffer
    // New buffer name will be like: [offset]_[name]_[offset]_[name]...
    auto& profilingOutputs = netOp.profilingOutputsInfo().front();
    SmallVector<uint32_t> outputBases;
    uint32_t totalSize = 0;
    std::string newOutputName;
    profilingOutputs.walk([&](IE::DataInfoOp op) {
        outputBases.push_back(totalSize);
        auto type = op.userType().cast<mlir::RankedTensorType>();
        newOutputName += std::to_string(totalSize) + '_' + op.name().str() + '_';
        auto size = static_cast<uint32_t>(type.getSizeInBits() / CHAR_BIT);
        totalSize += size;
        op.erase();
    });
    newOutputName.pop_back();

    //
    // Create and add new combined profiling output to the user info
    //
    auto newOutputResult =
            mlir::MemRefType::get({static_cast<int64_t>(totalSize / sizeof(uint32_t))}, getUInt32Type(ctx));
    auto newOutputShapedType = newOutputResult.cast<vpux::NDTypeInterface>();
    auto outputUserResult = getTensorType(newOutputShapedType.getShape(), newOutputShapedType.getElementType(),
                                          newOutputShapedType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputs.front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(
            mlir::NameLoc::get(mlir::Identifier::get("combinedProfilingDataOutputInfo", ctx)),
            mlir::StringAttr::get(ctx, newOutputName), mlir::TypeAttr::get(outputUserResult));

    auto totalArgumentsCount = netFunc.getNumArguments();
    auto mainArgumentsCount = totalArgumentsCount - outputBases.size();
    VPUX_THROW_UNLESS(mainArgumentsCount > 0, "There is no main network arguments in the funcOp");

    // Adding new output buffer
    auto newProfilngResult = netFunc.getBody().front().addArgument(newOutputResult);
    SmallVector<unsigned> argsToErase;
    unsigned removedArgs = 0;
    // Loop thought the old function arguments in order to replace its usages
    // with DeclareBufferOp with previously calculated offset connected to the new buffer
    // and erase these arguments
    auto arg = netFunc.getArguments().begin();
    while (arg != netFunc.getArguments().end()) {
        auto argNum = arg->getArgNumber();
        if (argNum > mainArgumentsCount - 1) {
            while (!arg->use_empty()) {
                auto use = arg->use_begin();
                auto op = use->getOwner();
                auto taskOp = op->getParentOfType<VPURT::TaskOp>();
                builder.setInsertionPoint((taskOp != nullptr) ? taskOp : op);
                unsigned base = (removedArgs) ? outputBases[removedArgs] : 0;
                auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                        mlir::NameLoc::get(mlir::Identifier::get("newProfilingBuffer", ctx)), arg->getType(),
                        VPURT::BufferSection::ProfilingOutput, 0, base);
                use->set(declareOp.buffer());
            }
            netFunc.eraseArgument(argNum);
            removedArgs++;
            // Reach the last old profiling output, stop here as the next output is the new buffer
            if (removedArgs == outputBases.size()) {
                break;
            }
        } else {
            arg++;
        }
    }

    //
    // Recalculate existing DeclareTensorOp
    //
    netFunc.walk([&](VPURT::DeclareBufferOp op) {
        if (op.section() == VPURT::BufferSection::ProfilingOutput) {
            auto sectionIndex = op.sectionIndex();
            if (sectionIndex.hasValue()) {
                VPUX_THROW_UNLESS(sectionIndex.getValue().size() == 1,
                                  "Profiling output is expected to have just one locale index");
                auto idx = parseIntArrayAttr<int64_t>(sectionIndex.getValue())[0];
                if (idx <= 0) {
                    return;
                }

                auto base = outputBases[idx];
                op.sectionIndexAttr(getIntArrayAttr(ctx, SmallVector<int64_t>({0})));
                auto offset = static_cast<uint32_t>(base + op.byteOffset());
                op.byteOffsetAttr(builder.getUI32IntegerAttr(offset));
            }
        }
    });

    //
    // Replace function signature
    //
    auto funcType = netFunc.getType();
    auto newResultTypes = to_small_vector(llvm::concat<const mlir::Type>(
            funcType.getResults().drop_back(outputBases.size()), makeArrayRef(newOutputResult)));
    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(newOutputResult)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);

    //
    // Replace function return operands
    //
    netFunc.walk([&](mlir::ReturnOp op) {
        auto start = static_cast<unsigned>(op.operandsMutable().size() - outputBases.size());
        op.operandsMutable().erase(start, static_cast<unsigned>(outputBases.size()));
        op.operandsMutable().append(newProfilngResult);
    });
}

}  // namespace

//
// createUPAProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUPAProfilingPass(Logger log) {
    return std::make_unique<UPAProfilingPass>(log);
}

//
// createGroupProfilingBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupProfilingBuffersPass(Logger log) {
    return std::make_unique<GroupProfilingBuffersPass>(log);
}
