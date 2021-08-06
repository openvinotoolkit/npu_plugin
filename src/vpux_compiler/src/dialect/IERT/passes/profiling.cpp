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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "mlir/IR/Attributes.h"

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

using namespace vpux;

namespace {

//
// AddLayoutsAndStridesPass
//

class TimestampProfilingPass final : public IERT::TimestampProfilingBase<TimestampProfilingPass> {
public:
    explicit TimestampProfilingPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

//
// safeRunOnModule
//

void TimestampProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    _memSpace = _memSpaceCb(ctx, "");
    if (_memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    auto* iert = getContext().getLoadedDialect<IERT::IERTDialect>();
    VPUX_THROW_UNLESS(iert != nullptr, "IERT Dialect was not loaded");
    const auto* layerInfo = iert->getRegisteredInterface<IERT::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "IERT Dialect was not initialized with LayerInfo interface");

    int dmaId = 0;
    SmallVector<IERT::TimestampOp> results;
    auto timestampType = mlir::MemRefType::get({1, 1, 1, 1}, getUInt32Type(ctx));

    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Process Operation '{0}'", op->getLoc());

        auto curTask = mlir::dyn_cast<RTLayerInterface>(op);
        if (curTask == nullptr) {
            _log.trace("It is not a VPUIP Task");
            return;
        }

        uint32_t curNumUnits = 0;
        const auto curExecutor = layerInfo->getExecutor(op, curNumUnits);
        auto physType = curExecutor.dyn_cast<VPUIP::PhysicalProcessorAttr>();
        if (physType == nullptr) {
            _log.trace("It is not a PhysicalProcessor Task");
            return;
        }

        builder.setInsertionPointAfter(op);
        int layerNumber = 0;
        std::string curTaskName = "[";
        curTaskName += curTask->getName().getStringRef().data();
        if ((physType.getValue() == VPUIP::PhysicalProcessor::NCE_Cluster) ||
            (physType.getValue() == VPUIP::PhysicalProcessor::NCE_PerClusterDPU)) {
            curTaskName += "_DPU]";
        } else {
            curTaskName += "_NA]";
        }

        if (const auto fusedLoc = curTask->getLoc().dyn_cast<mlir::FusedLoc>()) {
            auto locs = fusedLoc.getLocations();
            VPUX_THROW_UNLESS(locs.size() > 0, "FusedLoc is emply");
            if (const auto name = locs[0].dyn_cast<mlir::NameLoc>())
                curTaskName += name.getName().strref().data();
        } else if (const auto loc = curTask->getLoc().dyn_cast<mlir::NameLoc>()) {
            curTaskName += loc.getName().strref().data();
        }
        auto name = mlir::NameLoc::get(mlir::Identifier::get(
                curTaskName + ((dmaId == 0) ? "_PROFBEGIN_0" : ("_PROFMIDDLE_" + std::to_string(dmaId - 1))) + "_" +
                        std::to_string(layerNumber),
                ctx));

        auto timestampOp = builder.create<IERT::TimestampOp>(name, timestampType);
        results.push_back(timestampOp);
        dmaId++;
    };
    netFunc.walk(callback);

    VPUX_THROW_UNLESS(results.size(), "No TimestampOp was added");

    int output_size = dmaId;
    auto cmxMemType = mlir::MemRefType::get({1, output_size, 1, 1}, getUInt32Type(ctx), {}, _memSpace);
    auto outputResult = mlir::MemRefType::get({1, output_size, 1, 1}, getUInt32Type(ctx));

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    auto memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx), cmxMemType);

    SmallVector<int64_t> svStrides(timestampType.getShape().size(), 1);
    SmallVector<mlir::Value> dmas;
    for (uint32_t id = 0; id < results.size(); id++) {
        builder.setInsertionPointAfter(results[id]);
        auto sub = builder.create<mlir::memref::SubViewOp>(mlir::NameLoc::get(mlir::Identifier::get("subview", ctx)),
                                                           memOp, SmallVector<int64_t>({0, id, 0, 0}),
                                                           timestampType.getShape(), svStrides);

        dmas.push_back(builder.create<IERT::CopyOp>(results[id].getLoc(), results[id].output(), sub).output());
    }

    auto concatview = builder.create<IERT::ConcatViewOp>(mlir::NameLoc::get(mlir::Identifier::get("concatview", ctx)),
                                                         dmas, memOp.memref());

    //
    // Declare and create additional output from network
    //
    auto funcType = netFunc.getType();
    auto newResultTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), makeArrayRef(outputResult)));
    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(outputResult)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);
    auto profilngResult = netFunc.getBody().front().addArgument(outputResult);

    auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get("profilingCMX2DDR", ctx));
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), profilngResult);

    // Adding output to the user info
    auto outputUserResult =
            getTensorType(outputResult.getShape(), outputResult.getElementType(), DimsOrder::fromType(outputResult));
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.outputsInfo().front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "profilingOutput"),
                                           mlir::TypeAttr::get(outputUserResult));

    // And to the returnOp
    netFunc.walk([&](mlir::ReturnOp op) {
        op.operandsMutable().append(outputOp.output());
    });
}

}  // namespace

//
// createTimestampProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createTimestampProfilingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<TimestampProfilingPass>(std::move(memSpaceCb), log);
}
