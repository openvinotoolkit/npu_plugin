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

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"

namespace vpux {
namespace VPUIP {

class NCESparsity final {
public:
    using BiasConverterCb = std::int32_t (*)(double);
    using PPEConverterCb = std::int32_t (*)(unsigned, unsigned);

    static const vpux::EnumMap<vpux::VPUIP::ArchKind, PPEConverterCb> ppeConvertersMap;
    static const vpux::EnumMap<vpux::VPUIP::ArchKind, BiasConverterCb> biasConvertersMap;

    static int64_t getBitPatternSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType);
    static int64_t getActivationWindowSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType,
                                           int64_t inputChannels);
    static std::vector<uint8_t> getFakeSparsity(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                mlir::Type elemType, int64_t inputChannels, int64_t outputChannels);

    static std::vector<std::int32_t> getWeightsTable(mlir::Type op_inElemType, mlir::Type op_outElemType,
                                                     std::int32_t weightPtrOffset, std::int32_t weightPtrStep,
                                                     std::int32_t sparsityPtrOffset, vpux::VPUIP::ArchKind arch,
                                                     std::int64_t OC, mlir::Type weightsElemType = nullptr,
                                                     mlir::Value bias = nullptr);
};

}  // namespace VPUIP
}  // namespace vpux
