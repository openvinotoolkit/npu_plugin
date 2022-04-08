//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace ConstantFusing {

using ConstantVector = SmallVector<std::pair<VPUIP::CopyOp, Const::DeclareOp>>;
using TilingOpVector = SmallVector<std::pair<VPURT::AllocDistributed, VPUIP::NCEClusterTilingOp>>;

constexpr StringLiteral constantsFused = "constantsFused";
constexpr int8_t numberOfConstantsToFuse = 4;

///
/// \brief Converts U8 to I32 Datatype
/// \param [in] inValues - U8 Input value Content Range
/// \param [out] outValues - Vector of I32 converted Values
/// \return void
///

void convertInputToI32(Const::details::ContentRange<uint8_t>& inValues, std::vector<int32_t>& outValues);

///
/// \brief Converts the passed constant to U8 and store in passed vector
/// \param [in] inValue - T input value
/// \param [out] outValues - Vector of U8 converted Values
/// \return void
///

template <typename T, enable_if_t<or_<std::is_same<float16, T>, std::is_same<int32_t, T>>::value, bool> = true>
void convertToU8(const T& inValue, std::vector<uint8_t>& outValues) {
    const auto inputU8 = reinterpret_cast<const uint8_t*>(&inValue);
    for (size_t i = 0; i < sizeof(inValue); ++i)
        outValues.push_back(*(inputU8 + i));
}

///
/// \brief Get underlying DeclareOp and CopyOp for passed constant
/// \param [in] constant - Constant to get DeclareOp and CopyOp for
/// \param [out] constCopyOp - copyOp if the DeclareOp is found
/// \return Const::DeclareOp when found
///

Const::DeclareOp getConstAndCopyOp(VPUIP::NCEClusterTaskOp nceOp, mlir::Value constant, VPUIP::CopyOp& constCopyOp);

///
/// \brief Get static offset for the constant
/// \param [in] constant - mlir::Value constant
/// \return int32_t offset for the constant
///

int32_t getOffsetForConstant(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant);

/// @brief Function creates a new distributed buffer type, used for creating a alloc op for distributed buffer
/// @param origDistType [in] Used to get the original Distribute Mode
/// @param declOp [in] constant used for reference
/// @param rewriter [in] rewriter
/// @return VPUIP::DistributedBufferType [out]

VPUIP::DistributedBufferType getDistributedBufferType(VPUIP::DistributedBufferType origDistType,
                                                      Const::DeclareOp constant, mlir::PatternRewriter& rewriter);

/// @brief Gets the CopyOp and DeclareOp for constants to be fused
/// @param nceOp [in] The NCECluster task to which the constant belongs
/// @param constant [in] Constant (mlir::Value) for which the copy and const op are needed
/// @param copyOp [out] stores the copyOp found
/// @param declareOp [out] stores the declareOp found
/// @param allocDistributed [out] stores the allocDistributed if found for future use
/// @param tilingOp [out] stores the top TilingOp if found for future use

void getCopyAndDeclareOpForFusion(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant, VPUIP::CopyOp& copyOp,
                                  Const::DeclareOp& declareOp, VPURT::AllocDistributed& allocDistributed,
                                  VPUIP::NCEClusterTilingOp& tilingOp);
}  // namespace ConstantFusing
}  // namespace vpux
