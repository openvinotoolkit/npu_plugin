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

#pragma once

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/mlir/attributes.hpp"

#include <mlir/IR/StandardTypes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/attributes/enums.hpp.inc>

//
// EnumTraits
//

namespace vpux {

template <>
struct EnumTraits<VPUIP::PhysicalProcessor> final {
    static auto getEnumValueName(VPUIP::PhysicalProcessor val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::PhysicalProcessor>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 && static_cast<uint64_t>(val) <=
                                   VPUIP::getMaxEnumValForPhysicalProcessor();
    }
};

template <>
struct EnumTraits<VPUIP::DMAEngine> final {
    static auto getEnumValueName(VPUIP::DMAEngine val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::DMAEngine>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 &&
               static_cast<uint64_t>(val) <= VPUIP::getMaxEnumValForDMAEngine();
    }
};

template <>
struct EnumTraits<VPUIP::PhysicalMemory> final {
    static auto getEnumValueName(VPUIP::PhysicalMemory val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::PhysicalMemory>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 && static_cast<uint64_t>(val) <=
                                   VPUIP::getMaxEnumValForPhysicalMemory();
    }
};

template <>
struct EnumTraits<VPUIP::ArchKind> final {
    static auto getEnumValueName(VPUIP::ArchKind val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::ArchKind>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 &&
               static_cast<uint64_t>(val) <= VPUIP::getMaxEnumValForArchKind();
    }
};

template <>
struct EnumTraits<VPUIP::MemoryLocation> final {
    static auto getEnumValueName(VPUIP::MemoryLocation val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::MemoryLocation>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 && static_cast<uint64_t>(val) <=
                                   VPUIP::getMaxEnumValForMemoryLocation();
    }
};

template <>
struct EnumTraits<VPUIP::TaskType> final {
    static auto getEnumValueName(VPUIP::TaskType val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::TaskType>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 &&
               static_cast<uint64_t>(val) <= VPUIP::getMaxEnumValForTaskType();
    }
};

template <>
struct EnumTraits<VPUIP::ExecutionFlag> final {
    static auto getEnumValueName(VPUIP::ExecutionFlag val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::ExecutionFlag>(valStr);
    }

    static bool isValidVal(int64_t val) {
        return val >= 0 &&
               VPUIP::symbolizeExecutionFlag(checked_cast<uint32_t>(val))
                       .hasValue();
    }
};

}  // namespace vpux

namespace vpux {
namespace VPUIP {

using PhysicalProcessorAttr = IntEnumAttr<PhysicalProcessor>;
using DMAEngineAttr = IntEnumAttr<DMAEngine>;
using PhysicalMemoryAttr = IntEnumAttr<PhysicalMemory>;
using ArchKindAttr = IntEnumAttr<ArchKind>;
using MemoryLocationAttr = IntEnumAttr<MemoryLocation>;
using TaskTypeAttr = IntEnumAttr<TaskType>;
using ExecutionFlagAttr = IntEnumAttr<ExecutionFlag>;

//
// MemoryLocation utilities
//

MemoryLocation getDefaultMemoryLocation(PhysicalMemory mem);
MemoryLocation getDefaultMemoryLocation(mlir::MemRefType memref);

PhysicalMemory getPhysicalMemory(MemoryLocation location);
PhysicalMemory getPhysicalMemory(mlir::MemRefType memref);

bool isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref);

}  // namespace VPUIP
}  // namespace vpux
