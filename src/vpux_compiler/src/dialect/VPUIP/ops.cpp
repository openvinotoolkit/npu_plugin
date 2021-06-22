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
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// VPUIPLayerInfo
//

namespace {

class VPUIPLayerInfo final : public IERT::LayerInfoDialectInterface {
public:
    using IERT::LayerInfoDialectInterface::LayerInfoDialectInterface;

public:
    mlir::Attribute getExecutor(mlir::Operation* op, uint32_t& numUnits) const final;
    mlir::LogicalResult isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const final;
};

mlir::Attribute VPUIPLayerInfo::getExecutor(mlir::Operation* op, uint32_t& numUnits) const {
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

    if (mlir::isa<IERT::ConvolutionOp, IERT::MaxPoolOp>(op)) {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        const auto compileMode = VPUIP::getCompilationMode(module);

        if (compileMode == VPUIP::CompilationMode::ReferenceHW && VPUIP::NCEInvariant::verifyOp(op).succeeded()) {
            return getPhysicalProcessor(VPUIP::PhysicalProcessor::NCE_Cluster);
        }
    }

    if (mlir::isa<IERT::CopyOp>(op)) {
        return getDMAEngine(VPUIP::DMAEngine::DMA_NN);
    }

    return getPhysicalProcessor(VPUIP::PhysicalProcessor::SHAVE_UPA);
}

template <class ConcreteOp>
bool isSupportedByNCE(ConcreteOp op) {
    return VPUIP::NCEInvariant::verifyOp(op).succeeded();
}

mlir::LogicalResult VPUIPLayerInfo::isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    auto compileMode = VPUIP::getCompilationMode(module);

#define CASE(_IERT_OP_, _VPUIP_OP_)                     \
    .Case<_IERT_OP_>([&](_IERT_OP_ op) {                \
        return _VPUIP_OP_::isSupportedLayout(op, info); \
    })

#define HW_OPS_CASE(_IERT_OP_, _VPUIP_OP_)                                                \
    .Case<_IERT_OP_>([&](_IERT_OP_ op) {                                                  \
        if (compileMode == VPUIP::CompilationMode::ReferenceHW && isSupportedByNCE(op)) { \
            return VPUIP::NCEClusterTaskOp::isSupportedLayout(op, info);                  \
        }                                                                                 \
        return _VPUIP_OP_::isSupportedLayout(op, info);                                   \
    })

    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(origOp) CASE(IERT::QuantizeOp, VPUIP::QuantCastUPAOp)
    CASE(IERT::DequantizeOp, VPUIP::QuantCastUPAOp)
    CASE(IERT::ConvertOp, VPUIP::ConvertUPAOp)
    CASE(IERT::CopyOp, VPUIP::UPADMAOp)
    CASE(IERT::SoftMaxOp, VPUIP::SoftMaxUPAOp)
    CASE(IERT::AvgPoolOp, VPUIP::PoolingUPAOp)
    HW_OPS_CASE(IERT::MaxPoolOp, VPUIP::PoolingUPAOp)
    HW_OPS_CASE(IERT::ConvolutionOp, VPUIP::ConvolutionUPAOp)
    CASE(IERT::GroupConvolutionOp, VPUIP::ConvolutionUPAOp)
    CASE(IERT::ReLUOp, VPUIP::ReLUUPAOp)
    CASE(IERT::SigmoidOp, VPUIP::SigmoidUPAOp)
    CASE(IERT::ClampOp, VPUIP::ClampUPAOp)
    CASE(IERT::EluOp, VPUIP::EluUPAOp)
    CASE(IERT::HSwishOp, VPUIP::HSwishUPAOp)
    CASE(IERT::TanhOp, VPUIP::TanhUPAOp)
    CASE(IERT::FakeQuantizeOp, VPUIP::FakeQuantizeUPAOp)
    CASE(IERT::PReluOp, VPUIP::PReluUPAOp)
    CASE(IERT::LeakyReluOp, VPUIP::LeakyReluUPAOp)
    CASE(IERT::AddOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::MultiplyOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::DivideOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::SquaredDifferenceOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::PowerOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::FloorModOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::MinimumOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::MaximumOp, VPUIP::EltwiseUPAOp)
    CASE(IERT::SwishOp, VPUIP::SwishUPAOp)
    CASE(IERT::GRNOp, VPUIP::GRNUPAOp)
    CASE(IERT::TileOp, VPUIP::PerAxisTileUPAOp)
    CASE(IERT::PerAxisTileOp, VPUIP::PerAxisTileUPAOp)
    CASE(IERT::NegativeOp, VPUIP::NegativeUPAOp)
    CASE(IERT::ROIPoolingOp, VPUIP::ROIPoolingUPAOp)
    CASE(IERT::FullyConnectedOp, VPUIP::ConvolutionUPAOp)
    CASE(IERT::DetectionOutputOp, VPUIP::DetectionOutputUPAOp)
    CASE(IERT::ScaleShiftOp, VPUIP::ScaleShiftUPAOp)
    CASE(IERT::TransposeOp, VPUIP::PermuteUPAOp)
    CASE(IERT::ReorderOp, VPUIP::PermuteUPAOp)
    CASE(IERT::CTCGreedyDecoderOp, VPUIP::CTCGreedyDecoderUPAOp)
    CASE(IERT::CTCGreedyDecoderSeqLenOp, VPUIP::CTCGreedyDecoderSeqLenUPAOp)
    CASE(IERT::PadOp, VPUIP::PadUPAOp)
    CASE(IERT::ExpOp, VPUIP::ExpUPAOp)
    CASE(IERT::InterpolateOp, VPUIP::InterpolateUPAOp)
    .Default([](mlir::Operation* unknownOp) -> mlir::LogicalResult {
        VPUX_THROW("Operation '{0}' does not support layout propagation", unknownOp->getName());
    });

#undef CASE
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
    registry.addDialectInterface<IERT::IERTDialect, VPUIPLayerInfo>();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_CLASSES
