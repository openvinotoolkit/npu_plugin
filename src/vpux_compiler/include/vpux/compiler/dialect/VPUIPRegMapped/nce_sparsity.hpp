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

#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"

namespace vpux {
namespace VPUIPRegMapped {

class NCESparsity final {
public:
    using GetBiasCb = vpux::FuncRef<double(std::int64_t)>;
    using BiasConverterCb = std::int32_t (*)(double);
    using PPEConverterCb = std::int32_t (*)(double);

    static const vpux::EnumMap<vpux::VPUIPRegMapped::ArchKind, PPEConverterCb> ppeConvertersMap;
    static const vpux::EnumMap<vpux::VPUIPRegMapped::ArchKind, BiasConverterCb> biasConvertersMap;

    static int64_t getBitPatternSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType);
    static int64_t getActivationWindowSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType,
                                           int64_t inputChannels);
    static std::vector<uint8_t> getFakeSparsity(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                mlir::Type elemType, int64_t inputChannels);
    static std::vector<std::int32_t> getWeightsTable(std::int64_t OC, GetBiasCb getBiasFP, std::int32_t weightPtrOffset,
                                                     std::int32_t weightPtrStep, std::int32_t sparsityPtrOffset,
                                                     vpux::VPUIPRegMapped::ArchKind arch,
                                                     mlir::Type inputType = nullptr, mlir::Type weightsType = nullptr,
                                                     mlir::Type outputType = nullptr);
};

}  // namespace VPUIPRegMapped
}  // namespace vpux
