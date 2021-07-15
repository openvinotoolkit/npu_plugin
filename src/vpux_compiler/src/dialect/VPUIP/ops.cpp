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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerInfo
//

class LayerInfo final : public IE::LayerInfoDialectInterface {
public:
    using IE::LayerInfoDialectInterface::LayerInfoDialectInterface;

public:
    bool isSupportedPostProcessing(mlir::Operation* origOp, mlir::Operation* postOp) const final;
    bool needToExpandChannels(mlir::Operation* origOp) const final;
    bool isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const final;
};

bool LayerInfo::isSupportedPostProcessing(mlir::Operation* origOp, mlir::Operation* postOp) const {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto compileMode = VPUIP::getCompilationMode(module);

    if (!mlir::isa<IE::ReLUOp, IE::ScaleShiftOp>(postOp)) {
        return false;
    }

#define HW_OPS_CASE(_IE_OP_)                                      \
    .Case<_IE_OP_>([&](_IE_OP_ op) {                              \
        if (compileMode == VPUIP::CompilationMode::ReferenceSW) { \
            return false;                                         \
        }                                                         \
        return VPUIP::NCEInvariant::verifyKernel(op).succeeded(); \
    })

    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)  //
            HW_OPS_CASE(IE::ConvolutionOp)                   //
    HW_OPS_CASE(IE::MaxPoolOp)                               //
    .Default([](mlir::Operation*) {
        return false;
    });

#undef HW_OPS_CASE
}

bool LayerInfo::needToExpandChannels(mlir::Operation* origOp) const {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto compileMode = VPUIP::getCompilationMode(module);

#define HW_OPS_CASE(_IE_OP_)                                         \
    .Case<_IE_OP_>([&](_IE_OP_ op) {                                 \
        if (compileMode == VPUIP::CompilationMode::ReferenceSW) {    \
            return false;                                            \
        }                                                            \
        if (VPUIP::NCEInvariant::verifyKernel(op).failed()) {        \
            return false;                                            \
        }                                                            \
        return !VPUIP::NCEInvariant::verifyChannels(op).succeeded(); \
    })

    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)  //
            HW_OPS_CASE(IE::ConvolutionOp)                   //
    HW_OPS_CASE(IE::MaxPoolOp)                               //
    HW_OPS_CASE(IE::AddOp)                                   //
    .Default([](mlir::Operation*) {
        return false;
    });

#undef HW_OPS_CASE
}

bool LayerInfo::isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto compileMode = VPUIP::getCompilationMode(module);

#define CASE(_IE_OP_, _VPUIP_OP_)                       \
    .Case<_IE_OP_>([&](_IE_OP_ op) {                    \
        return _VPUIP_OP_::isSupportedLayout(op, info); \
    })

#define HW_OPS_CASE(_IE_OP_, _VPUIP_OP_)                             \
    .Case<_IE_OP_>([&](_IE_OP_ op) {                                 \
        if (compileMode == VPUIP::CompilationMode::ReferenceSW) {    \
            return _VPUIP_OP_::isSupportedLayout(op, info);          \
        }                                                            \
        if (VPUIP::NCEInvariant::verifyKernel(op).failed()) {        \
            return _VPUIP_OP_::isSupportedLayout(op, info);          \
        }                                                            \
        if (VPUIP::NCEInvariant::verifyChannels(op).failed()) {      \
            return _VPUIP_OP_::isSupportedLayout(op, info);          \
        }                                                            \
        return VPUIP::NCEClusterTaskOp::isSupportedLayout(op, info); \
    })

    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)  //
            CASE(IE::ConvertOp, VPUIP::ConvertUPAOp)
    CASE(IE::SoftMaxOp, VPUIP::SoftMaxUPAOp)
    CASE(IE::AvgPoolOp, VPUIP::PoolingUPAOp)
    HW_OPS_CASE(IE::MaxPoolOp, VPUIP::PoolingUPAOp)
    HW_OPS_CASE(IE::ConvolutionOp, VPUIP::ConvolutionUPAOp)
    HW_OPS_CASE(IE::AddOp, VPUIP::EltwiseUPAOp)
    CASE(IE::GroupConvolutionOp, VPUIP::ConvolutionUPAOp)
    CASE(IE::ReLUOp, VPUIP::ReLUUPAOp)
    CASE(IE::SigmoidOp, VPUIP::SigmoidUPAOp)
    CASE(IE::ClampOp, VPUIP::ClampUPAOp)
    CASE(IE::EluOp, VPUIP::EluUPAOp)
    CASE(IE::HSwishOp, VPUIP::HSwishUPAOp)
    CASE(IE::TanhOp, VPUIP::TanhUPAOp)
    CASE(IE::FakeQuantizeOp, VPUIP::FakeQuantizeUPAOp)
    CASE(IE::PReluOp, VPUIP::PReluUPAOp)
    CASE(IE::LeakyReluOp, VPUIP::LeakyReluUPAOp)
    CASE(IE::MultiplyOp, VPUIP::EltwiseUPAOp)
    CASE(IE::DivideOp, VPUIP::EltwiseUPAOp)
    CASE(IE::SquaredDifferenceOp, VPUIP::EltwiseUPAOp)
    CASE(IE::PowerOp, VPUIP::EltwiseUPAOp)
    CASE(IE::FloorModOp, VPUIP::EltwiseUPAOp)
    CASE(IE::MinimumOp, VPUIP::EltwiseUPAOp)
    CASE(IE::MaximumOp, VPUIP::EltwiseUPAOp)
    CASE(IE::SwishOp, VPUIP::SwishUPAOp)
    CASE(IE::GRNOp, VPUIP::GRNUPAOp)
    CASE(IE::TileOp, VPUIP::PerAxisTileUPAOp)
    CASE(IE::PerAxisTileOp, VPUIP::PerAxisTileUPAOp)
    CASE(IE::NegativeOp, VPUIP::NegativeUPAOp)
    CASE(IE::ROIPoolingOp, VPUIP::ROIPoolingUPAOp)
    CASE(IE::FullyConnectedOp, VPUIP::FullyConnectedUPAOp)
    CASE(IE::DetectionOutputOp, VPUIP::DetectionOutputUPAOp)
    CASE(IE::ScaleShiftOp, VPUIP::ScaleShiftUPAOp)
    CASE(IE::TransposeOp, VPUIP::PermuteUPAOp)
    CASE(IE::ReorderOp, VPUIP::PermuteUPAOp)
    CASE(IE::CTCGreedyDecoderOp, VPUIP::CTCGreedyDecoderUPAOp)
    CASE(IE::CTCGreedyDecoderSeqLenOp, VPUIP::CTCGreedyDecoderSeqLenUPAOp)
    CASE(IE::PadOp, VPUIP::PadUPAOp)
    CASE(IE::ExpOp, VPUIP::ExpUPAOp)
    CASE(IE::InterpolateOp, VPUIP::InterpolateUPAOp)
    CASE(IE::StridedSliceOp, VPUIP::StridedSliceUPAOp)
    .Default([](mlir::Operation* unknownOp) -> bool {
        VPUX_THROW("Operation '{0}' does not support layout propagation", unknownOp->getName());
    });

#undef CASE
#undef HW_OPS_CASE
}

//
// RTLayerInfo
//

class RTLayerInfo final : public IERT::LayerInfoDialectInterface {
public:
    using IERT::LayerInfoDialectInterface::LayerInfoDialectInterface;

public:
    mlir::Attribute getExecutor(mlir::Operation* op, uint32_t& numUnits) const final;
};

mlir::Attribute RTLayerInfo::getExecutor(mlir::Operation* op, uint32_t& numUnits) const {
    const auto getDMAEngine = [&](VPUIP::DMAEngine engine) {
        numUnits = 1;
        return VPUIP::DMAEngineAttr::get(op->getContext(), engine);
    };

    const auto getPhysicalProcessor = [&](VPUIP::PhysicalProcessor proc, Optional<uint32_t> units = None) {
        const auto procAttr = VPUIP::PhysicalProcessorAttr::get(op->getContext(), proc);

        if (units.hasValue()) {
            numUnits = units.getValue();
        } else {
            auto module = op->getParentOfType<mlir::ModuleOp>();
            auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
            auto available = resources.getExecutor(procAttr);
            VPUX_THROW_UNLESS(available != nullptr, "Executor for '{0}' is not available", procAttr);
            numUnits = available.count();
        }

        return procAttr;
    };

    if (auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op)) {
        const auto taskType = task.getTaskType();

        switch (taskType) {
        case VPUIP::TaskType::UPADMA:
            return getDMAEngine(VPUIP::DMAEngine::DMA_UPA);
        case VPUIP::TaskType::NNDMA:
            return getDMAEngine(VPUIP::DMAEngine::DMA_NN);
        case VPUIP::TaskType::NCE2:
            return getPhysicalProcessor(VPUIP::PhysicalProcessor::NCE_Cluster, 1);
        case VPUIP::TaskType::UPA: {
            auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
            return getPhysicalProcessor(VPUIP::PhysicalProcessor::SHAVE_UPA, upaTask.maxShaves());
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskType);
        }
    }

    if (mlir::isa<IERT::CopyOp>(op)) {
        return getDMAEngine(VPUIP::DMAEngine::DMA_NN);
    }

    return getPhysicalProcessor(VPUIP::PhysicalProcessor::SHAVE_UPA);
}

}  // namespace

//
// initialize
//

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_LIST
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();
}

//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addDialectInterface<IE::IEDialect, LayerInfo>();
    registry.addDialectInterface<IERT::IERTDialect, RTLayerInfo>();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
