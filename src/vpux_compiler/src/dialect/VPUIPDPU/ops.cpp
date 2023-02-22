//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <functional>

using namespace vpux;
using namespace vpux::VPUIPDPU;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPDPU/generated/ops.cpp.inc>

//
// Custom
//

namespace {

template <typename OpType>
int countEntryBlockOps(Operation* operation) {
    if (!operation->getNumRegions())
        return 0;

    auto& blocks = operation->getRegion(0).getBlocks();
    if (blocks.empty())
        return 0;

    auto ops = blocks.front().getOps<OpType>();
    return checked_cast<int>(std::distance(ops.begin(), ops.end()));
}

template <typename ChildOpType, typename ParentOpType>
bool validateChildOp(ParentOpType& parentOp, std::function<bool(int)> isValid) {
    static_assert(ChildOpType::template hasTrait<::mlir::OpTrait::template HasParent<ParentOpType>::template Impl>(),
                  "ChildOpType does not belong to ParentOpType");

    return isValid(countEntryBlockOps<ChildOpType>(parentOp.getOperation()));
}

// base recursion function - needed as recursion terminator
template <typename ParentOpType>
bool checkChildrenInstances(ParentOpType&, std::function<bool(int)>) {
    return true;
}

template <typename ParentOpType, typename FirstChildOpType, typename... RemainingChildOpTypes>
bool checkChildrenInstances(ParentOpType& p, std::function<bool(int)> isValid) {
    if (!validateChildOp<FirstChildOpType>(p, isValid)) {
        return false;
    }

    return checkChildrenInstances<ParentOpType, RemainingChildOpTypes...>(p, isValid);
}

template <typename ParentOpType, typename... ChildOpTypes>
bool checkMandatoryChildren(ParentOpType& parentOp) {
    return checkChildrenInstances<ParentOpType, ChildOpTypes...>(parentOp, [](int numChildOpInstances) {
        return numChildOpInstances == 1;
    });
}

template <typename ParentOpType, typename... ChildOpTypes>
bool checkOptionalChildren(ParentOpType& parentOp) {
    return checkChildrenInstances<ParentOpType, ChildOpTypes...>(parentOp, [](int numChildOpInstances) {
        return (numChildOpInstances & ~1) == 0;  // allow maximum 1 child instance
    });
}

mlir::LogicalResult verifyDPUInvariant(DPUInvariant& op) {
    if (!checkMandatoryChildren<DPUInvariant, IDUCfgOp, PPECfgOp, ODUCfgOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: missing mandatory child ops", op.getOperationName());
    }
    if (!checkOptionalChildren<DPUInvariant, MPECfgOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult verifyDPUVariant(DPUVariant& op) {
    if (!checkMandatoryChildren<DPUVariant, ODUOutSubtensorOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: missing mandatory child ops", op.getOperationName());
    }

    return ::mlir::success();
}

template <typename T>
mlir::LogicalResult verifyPPEBiasAdd(T& op) {
    auto scaleTableExists = (op.scale_table() != nullptr);
    auto biasStaticExists = op.bias_static().hasValue();

    // scale_table only
    if (scaleTableExists && !biasStaticExists) {
        return ::mlir::success();
    }

    // bias_static only
    if (biasStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0} needs either scale_table or bias_static as parameter",
                   op.getOperationName());
}

template <typename T>
mlir::LogicalResult verifyPPEScaleMult(T& op) {
    auto scaleTableExists = (op.scale_table() != nullptr);
    auto scaleStaticExists = op.scale_static().hasValue();

    // scale_table only
    if (scaleTableExists && !scaleStaticExists) {
        return ::mlir::success();
    }

    // scale_static only
    if (scaleStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0} needs either scale_table or scale_static as parameter",
                   op.getOperationName());
}

mlir::LogicalResult verifyPPEFpConv(PPEFpConvertOp& op) {
    auto convMode = op.convert_mode();
    auto clampModeExists = op.clamp_mode().hasValue();
    auto ftzModeExists = op.ftz_mode().hasValue();
    auto bf16RoundModeExists = op.bf16_round_mode().hasValue();

    // Validate the supported conversions with different clamp, ftz and bf16 combinations
    switch (convMode) {
    case PPEFpConvertMode::FpConv_Fp16_RNE:
        if (clampModeExists && ftzModeExists && !bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::FpConv_Bfp16:
        if (!clampModeExists && !ftzModeExists && bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::FpConv_Fp8_RNE:
        if (clampModeExists && ftzModeExists && !bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::FpConv_Bypass:
    case PPEFpConvertMode::FpConv_I32_RNE:
        if (!clampModeExists && !ftzModeExists && !bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    default:
        break;
    }

    return errorAt(op.getLoc(), "Operation {0} has unsupported combination of parameters", op.getOperationName());
}

mlir::LogicalResult verifyPPEScaleShift(PPEIntScaleShiftOp& op) {
    auto scaleTableExists = (op.scale_table() != nullptr);
    auto shiftStaticExists = op.shift_static().hasValue();

    // scale_table only
    if (scaleTableExists && !shiftStaticExists) {
        return ::mlir::success();
    }

    // scale_static only
    if (shiftStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0} needs either scale_table or shift_static as parameter",
                   op.getOperationName());
}

mlir::LogicalResult verifyODUCfg(ODUCfgOp& op) {
    if (!checkMandatoryChildren<ODUCfgOp, ODUOutTensorSizeOp, ODUOutActivationsOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: missing mandatory child ops", op.getOperationName());
    }

    if (!checkOptionalChildren<ODUCfgOp, ODUDataReuseOp, ODUPermuteDataOp, ODUSparsityOp, ODUSwizzleDataOp,
                               ODUMemoryModeOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    // VPU2.7 supports up to 3 cast instances
    if (countEntryBlockOps<ODUCastOp>(op.getOperation()) > 3) {
        return errorAt(op.getLoc(), "Operation {0}: too many cast instances defined", op.getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult verifyODUSparsity(ODUSparsityOp& op) {
    auto sparsityMapExists = (op.sparsity_map() != nullptr);
    auto compressionEnabledExists = op.compression_enabled().hasValue();
    bool compressionEnabled = false;
    if (compressionEnabledExists) {
        compressionEnabled = op.compression_enabled().getValue();
    }
    auto sparseValueExists = op.sparse_value().hasValue();

    if (!sparsityMapExists && compressionEnabledExists && compressionEnabled) {
        return ::mlir::success();
    }

    if (sparsityMapExists && !compressionEnabledExists) {
        return ::mlir::success();
    }

    if (sparsityMapExists && compressionEnabledExists && !compressionEnabled && !sparseValueExists) {
        return ::mlir::success();
    }

    if (sparsityMapExists && compressionEnabledExists && compressionEnabled) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0}: invalid params combination", op.getOperationName());
}

mlir::LogicalResult verifyODUOutActivations(ODUOutActivationsOp& op) {
    auto arch = VPU::getArch(op);
    auto dataTypeExists = op.data_type().hasValue();

    if (!dataTypeExists) {
        return ::mlir::success();
    }

    if ((arch == VPU::ArchKind::VPUX37XX) && !dataTypeExists) {
        return errorAt(op.getLoc(), "Operation {0}: use data_type attr to specify data type", op.getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult verifyMPECfg(MPECfgOp& op) {
    if (!checkOptionalChildren<MPECfgOp, MPEDenormalOperandsFTZOp, MPEActivationBiasOp, MPEWeightsBiasOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    return ::mlir::success();
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult verifyOp(DPUInvariant op) {
    return verifyDPUInvariant(op);
}

mlir::LogicalResult verifyOp(DPUVariant op) {
    return verifyDPUVariant(op);
}

mlir::LogicalResult verifyOp(PPEFpBiasAddOp op) {
    return verifyPPEBiasAdd(op);
}

mlir::LogicalResult verifyOp(PPEFpScaleMultOp op) {
    return verifyPPEScaleMult(op);
}

mlir::LogicalResult verifyOp(PPEFpConvertOp op) {
    return verifyPPEFpConv(op);
}

mlir::LogicalResult verifyOp(PPEIntBiasAddOp op) {
    return verifyPPEBiasAdd(op);
}

mlir::LogicalResult verifyOp(PPEIntScaleMultOp op) {
    return verifyPPEScaleMult(op);
}

mlir::LogicalResult verifyOp(PPEIntScaleShiftOp op) {
    return verifyPPEScaleShift(op);
}

mlir::LogicalResult verifyOp(ODUCfgOp op) {
    return verifyODUCfg(op);
}

mlir::LogicalResult verifyOp(ODUSparsityOp op) {
    return verifyODUSparsity(op);
}

mlir::LogicalResult verifyOp(ODUOutActivationsOp op) {
    return verifyODUOutActivations(op);
}

mlir::LogicalResult verifyOp(MPECfgOp op) {
    return verifyMPECfg(op);
}

}  // namespace VPUIPDPU
}  // namespace vpux
