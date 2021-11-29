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

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

class NCESparsity final {
public:
    using BiasConverterCb = int32_t (*)(double);
    using PPEConverterCb = int32_t (*)(unsigned, unsigned, double, mlir::Type);

    static const EnumMap<VPU::ArchKind, PPEConverterCb> ppeConvertersMap;
    static const EnumMap<VPU::ArchKind, BiasConverterCb> biasConvertersMap;

    static int64_t getBitPatternSize(ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType);
    static int64_t getActivationWindowSize(ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType,
                                           int64_t inputChannels);
    static std::vector<uint8_t> getFakeSparsity(ArrayRef<int64_t> kernelSize, int64_t strideW, mlir::Type elemType,
                                                int64_t inputChannels);

    static std::vector<int32_t> getWeightsTable(mlir::Type op_inElemType, mlir::Type op_outElemType,
                                                Optional<int32_t> weightPtrOffset, int32_t weightPtrStep,
                                                Optional<int32_t> sparsityPtrOffset, VPU::ArchKind arch, int64_t OC,
                                                mlir::Type weightsElemType = nullptr, mlir::Value bias = nullptr);
};

}  // namespace VPUIP
}  // namespace vpux
