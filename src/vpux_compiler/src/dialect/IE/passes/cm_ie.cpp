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

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>
#include <numeric>


using namespace vpux;

namespace {

//
// Optimizer
//

class ChannelMajorConvolutionCompatibleOps final {
public:
    explicit ChannelMajorConvolutionCompatibleOps (mlir::MLIRContext* ctx, Logger log, vpux::DimsOrder userDimsOrder): 
     _ctx(ctx), _log(log), _userDimsOrder(userDimsOrder) {
        Logger::global().error("order: {0}", _userDimsOrder);

    }

    void identifyCompatibleChannelMajorOps(mlir::ModuleOp module, Logger log);

private:
    
    vpux::DimsOrder _userDimsOrder;
    SmallVector<mlir::Operation*> _allConvOps;
    Logger _log;
    mlir::MLIRContext* _ctx;
 
};

void ChannelMajorConvolutionCompatibleOps::identifyCompatibleChannelMajorOps(mlir::ModuleOp module, Logger log) {

    for (mlir::Operation& op : module) {
        if (mlir::isa<mlir::FuncOp>(op)) {
            auto func = mlir::dyn_cast<mlir::FuncOp>(op);
            for (auto& op : func.getOps())
                if (auto convOp = mlir::dyn_cast<vpux::IE::ConvolutionOp>(op)) {
                    const auto inputShape = getShape(convOp.filter().getType().cast<mlir::ShapedType>());
                    const auto IC = inputShape[IE::Dims4D::Filter::IC];

                    auto inputTensorShape = getShape(convOp.input());
                    auto width = inputTensorShape[IE::Dims4D::Act::W];

                    Logger::global().error("order: {0}", IC);
                    Logger::global().error("order: {0}", width);

                    if((IC == 3) && (width%16 == 0) && _userDimsOrder == DimsOrder::NCHW)
                    {
                         Logger::global().error("CM");
                         convOp->setAttr("ChannelMajorCompitable", getIntAttr(_ctx, 1));
                         convOp->getAttr("ChannelMajorCompitable").cast<mlir::IntegerAttr>().getInt();
                         Logger::global().error("ChannelMajorCompitable: {0}", convOp->getAttr("ChannelMajorCompitable").cast<mlir::IntegerAttr>().getInt());
                    }
                    else
                         convOp->setAttr("ChannelMajorCompitable", getIntAttr(_ctx, 0));
                }
        }
    }
}

class ChannelMajorConvolutionCompatibleOpsPass final : public IE::ChannelMajorConvolutionCompatibleOpsBase<ChannelMajorConvolutionCompatibleOpsPass> {
public:
    explicit ChannelMajorConvolutionCompatibleOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ChannelMajorConvolutionCompatibleOpsPass::safeRunOnModule() {

    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    const auto funcType = netFunc.getType();

    auto userInputs = netInfo.getInputsInfo();
    auto userOutputs = netInfo.getOutputsInfo();
    vpux::DimsOrder userDimsOrder; 

    const auto getTypesWithUserLayout = [](SmallVector<IE::DataInfoOp, 1>& userDataInfo,
                                           ArrayRef<mlir::Type> originTypes, SmallVector<mlir::Type>& newTypes, vpux::DimsOrder& userDimsOrder) {
        for (const auto& p : userDataInfo | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());

            const auto origType = originTypes[ind].cast<mlir::ShapedType>();
            userDimsOrder = p.value().getDimsOrder();
            Logger::global().error("order: {0}", userDimsOrder);

        }
    };

    SmallVector<mlir::Type> newArgTypes(userInputs.size());
    getTypesWithUserLayout(userInputs, funcType.getInputs(), newArgTypes, userDimsOrder);

    Logger::global().error("order: {0}", userDimsOrder);
    ChannelMajorConvolutionCompatibleOps cmconv(ctx, _log, userDimsOrder);
    cmconv.identifyCompatibleChannelMajorOps(module, _log);


    // if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
    //     signalPassFailure();
    // }

   
}

}  // namespace

//
// createOptimizeAsyncDepsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createChannelMajorConvolutionCompatibleOpsPass(Logger log) {
    return std::make_unique<ChannelMajorConvolutionCompatibleOpsPass>(log);
}