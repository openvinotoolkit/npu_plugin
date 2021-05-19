//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

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
    return llvm::TypeSwitch<mlir::Operation*, mlir::Attribute>(op)
            .Case<IERT::CopyOp>([&](IERT::CopyOp) {
                numUnits = 1;
                return VPUIP::DMAEngineAttr::get(op->getContext(), VPUIP::DMAEngine::DMA_NN);
            })
            .Default([&](mlir::Operation*) {
                auto module = op->getParentOfType<mlir::ModuleOp>();
                auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
                auto upaSHAVEs = resources.getExecutor(
                        VPUIP::PhysicalProcessorAttr::get(op->getContext(), VPUIP::PhysicalProcessor::SHAVE_UPA));
                numUnits = upaSHAVEs.count();
                return upaSHAVEs.kind();
            });
}

mlir::LogicalResult VPUIPLayerInfo::isSupportedLayout(mlir::Operation* origOp, DataOrderInfo& info) const {
#define CASE(_IERT_OP_, _VPUIP_OP_)                     \
    .Case<_IERT_OP_>([&](mlir::Operation* op) {         \
        return _VPUIP_OP_::isSupportedLayout(op, info); \
    })

    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(origOp) CASE(IERT::QuantizeOp, VPUIP::QuantCastUPAOp)
    CASE(IERT::DequantizeOp, VPUIP::QuantCastUPAOp)
    CASE(IERT::ConvertOp, VPUIP::ConvertUPAOp)
    CASE(IERT::CopyOp, VPUIP::UPADMAOp)
    CASE(IERT::SoftMaxOp, VPUIP::SoftMaxUPAOp)
    CASE(IERT::AvgPoolOp, VPUIP::PoolingUPAOp)
    CASE(IERT::MaxPoolOp, VPUIP::PoolingUPAOp)
    CASE(IERT::ConvolutionOp, VPUIP::ConvolutionUPAOp)
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
