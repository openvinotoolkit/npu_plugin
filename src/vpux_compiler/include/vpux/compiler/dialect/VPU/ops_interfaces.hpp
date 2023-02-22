//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

//
// SparseOpInterface
//

bool supportsSparseInputs(mlir::Operation* op);
bool supportsSparseOutputs(mlir::Operation* op);
bool supportsSparseData(mlir::Operation* op);
bool supportsSparseWeights(mlir::Operation* op);

//
// NCEOpInterface
//

template <typename ConcreteOp>
void setLayerMultiClusterStrategyAttr(ConcreteOp mainOp, VPU::MultiClusterStrategy strategy) {
    const auto multiClusterStrategyAttr = VPU::MultiClusterStrategyAttr::get(mainOp->getContext(), strategy);
    mainOp.multiClusterStrategyAttr(multiClusterStrategyAttr);
}

namespace details {

mlir::LogicalResult validatePrecisionForNCE(mlir::Operation* op);
mlir::LogicalResult validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads);

mlir::Operation* addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc, ShapeRef offsets,
                             ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode, mlir::IntegerAttr clusterId);

}  // namespace details

//
// TilingBuilderOpInterface
//

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const TileInfo& tile,
                     StringRef valName);

//
// TilingInfoOpInterface
//

mlir::LogicalResult verifyTilingInfo(mlir::Operation* op);

//
// EltwiseOp
//

mlir::LogicalResult verifyEltwiseOp(mlir::Operation* op);

template <typename ConcreteOp>
class EltwiseOp : public mlir::OpTrait::TraitBase<ConcreteOp, EltwiseOp> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return VPU::verifyEltwiseOp(op);
    }

    InputTiling backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
        return backInferEltwiseTile(this->getOperation(), outputTile);
    }

    void adjustAttrs(const TilingInfo&, const TileInfo&) {
        // Do nothing
    }

    OutputTiling getTilingStrategy(TilingMode tilingMode, Logger log) {
        return getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
    }
};

//
// NCEOpInterface
//

mlir::LogicalResult verifyNCEOp(mlir::Operation* op);

//
// isPureViewOp
//

bool isPureViewOp(mlir::Operation* op);

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDimsOrder(IE::LayerLayoutInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutDimsOrder(info);
    }
};

//
// SameInOutDimsOrder_NCHW_NHWC
//

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
void inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info, ArrayRef<DimsOrder> supportedLayouts);

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_NHWC : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW, DimsOrder::NHWC});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::NHWC});
    }
};

//
// SameInOutDimsOrder_NCHW
//

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    }
};

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

template <typename ConcreteOp>
class SameInOutDimsOrder_CHW_HWC_NCHW_NHWC :
        public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_CHW_HWC_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(info,
                                                  {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
};

//
// SameInOutDimsOrder_NCHW_CHW_NC_C
//

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_CHW_NC_C : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_CHW_NC_C> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW, DimsOrder::CHW, DimsOrder::NC, DimsOrder::C});
    }
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        // [Track number: E#25740]
        inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::CHW, DimsOrder::NC, DimsOrder::C});
    }
};

//
// AnyDimsOrder
//

template <typename ConcreteOp>
class AnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, AnyDimsOrder> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo&) {
    }
};

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.hpp.inc>
