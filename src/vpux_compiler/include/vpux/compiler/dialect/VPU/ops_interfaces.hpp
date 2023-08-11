//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>

#include <initializer_list>
#include <numeric>

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
// SameInOutDefaultDimsOrder
//

mlir::LogicalResult verifySameInOutDefaultDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDefaultDimsOrder(IE::LayerLayoutInfo& info);

template <typename ConcreteOp>
class SameInOutDefaultDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDefaultDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDefaultDimsOrder(op);
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutDefaultDimsOrder(info);
    }
};

//
// SameAnyDimsOrder
//

mlir::LogicalResult verifySameAnyDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameAnyDimsOrder(IE::LayerLayoutInfo& info);

template <typename ConcreteOp>
class SameAnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameAnyDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameAnyDimsOrder(op);
    }
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameAnyDimsOrder(info);
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
// SameInOutDimsOrder_NC_CHW_HWC_NCHW_NHWC
//

template <typename ConcreteOp>
class SameInOutDimsOrder_NC_CHW_HWC_NCHW_NHWC :
        public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NC_CHW_HWC_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(
                op, {DimsOrder::NC, DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::NC, DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
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

//
// LimitedToArch
//

template <ArchKind... archs>
struct LimitedToArch {
    template <typename ConcreteType>
    class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    public:
        static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
            return verifyArchKind(op, {archs...});
        }

    private:
        static mlir::LogicalResult verifyArchKind(mlir::Operation* op, std::initializer_list<ArchKind> supportedArchs) {
            auto arch = getArch(op);

            if (arch != ArchKind::UNKNOWN) {
                if (std::find(cbegin(supportedArchs), cend(supportedArchs), arch) == cend(supportedArchs)) {
                    auto archStr = stringifyArchKind(arch).str();
                    auto archsStr = std::accumulate(cbegin(supportedArchs), cend(supportedArchs), std::string(),
                                                    [](const std::string& accu, const ArchKind arch) -> std::string {
                                                        return accu + (accu.length() > 0 ? "," : "") +
                                                               stringifyArchKind(arch).str();
                                                    });
                    return vpux::errorAt(op, "Operation {0} not supported in {1}; list of supported archs: {2}",
                                         op->getName(), archStr, archsStr);
                }
            }

            return mlir::success();
        }
    };
};

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.hpp.inc>
