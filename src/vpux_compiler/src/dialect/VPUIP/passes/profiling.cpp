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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/strings.hpp"

using namespace vpux;

namespace {

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

    auto& profilingOutputs = netOp.profilingOutputsInfo().front();
    SmallVector<uint32_t> outputBases;
    uint32_t total_size = 0;
    std::string newOutputName;
    profilingOutputs.walk([&](IE::DataInfoOp op) {
        outputBases.push_back(total_size);
        auto type = op.userType().cast<mlir::RankedTensorType>();
        newOutputName += std::to_string(total_size) + '_' + op.name().str() + '_';
        auto size = type.getSizeInBits() / 8;
        total_size += size;
        op.erase();
    });
    newOutputName.pop_back();
    std::cout << newOutputName << "\n";

    // Create new combined profiling output to the user info
    auto newOutputResult = mlir::MemRefType::get({total_size / 4}, getUInt32Type(ctx));
    auto outputUserResult = getTensorType(getShape(newOutputResult), newOutputResult.getElementType(),
                                          DimsOrder::fromType(newOutputResult), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputs.front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, newOutputName),
                                           mlir::TypeAttr::get(outputUserResult));

    auto totalArgumentsCount = netFunc.getNumArguments();
    auto mainArgumentsCount = totalArgumentsCount - outputBases.size();
    auto newProfilngResult = netFunc.getBody().front().addArgument(newOutputResult);
    SmallVector<unsigned> argsToErase;
    unsigned removedArgs = 0;
    auto arg = netFunc.getArguments().begin();
    while (arg != netFunc.getArguments().end()) {
        auto argNum = arg->getArgNumber();
        if (argNum > mainArgumentsCount - 1) {
            while (!arg->use_empty()) {
                auto use = arg->use_begin();
                auto op = use->getOwner();
                builder.setInsertionPoint(op);
                unsigned base = (removedArgs) ? outputBases[removedArgs] : 0;
                auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                        mlir::UnknownLoc::get(ctx), arg->getType(), VPUIP::MemoryLocation::ProfilingOutput, 0, base);
                use->set(declareOp.memory());
            }
            netFunc.eraseArgument(argNum);
            removedArgs++;
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
        if (op.locale() == VPUIP::MemoryLocation::ProfilingOutput) {
            auto localeIndex = parseIntArrayAttr<uint32_t>(op.localeIndex());
            if (localeIndex[0] > 0) {
                auto base = outputBases[localeIndex[0]];
                op.localeIndexAttr(builder.getI64ArrayAttr(ArrayRef<int64_t>{0}));
                auto offset = base + op.dataIndex();
                op.dataIndexAttr(builder.getUI32IntegerAttr(offset));
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
        auto start = op.operandsMutable().size() - outputBases.size();
        op.operandsMutable().erase(start, outputBases.size());
        op.operandsMutable().append(newProfilngResult);
    });
}

}  // namespace

//
// createGroupProfilingBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupProfilingBuffersPass(Logger log) {
    return std::make_unique<GroupProfilingBuffersPass>(log);
}
