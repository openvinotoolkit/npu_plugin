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
};

template <>
struct EnumTraits<VPUIP::DMAEngine> final {
    static auto getEnumValueName(VPUIP::DMAEngine val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::DMAEngine>(valStr);
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
};

template <>
struct EnumTraits<VPUIP::ArchKind> final {
    static auto getEnumValueName(VPUIP::ArchKind val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::ArchKind>(valStr);
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
};

template <>
struct EnumTraits<VPUIP::ExecutionFlag> final {
    static auto getEnumValueName(VPUIP::ExecutionFlag val) {
        return VPUIP::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return VPUIP::symbolizeEnum<VPUIP::ExecutionFlag>(valStr);
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
};

}  // namespace vpux

namespace vpux {
namespace VPUIP {

//
// PhysicalProcessorAttr
//

class PhysicalProcessorAttr final
        : public EnumAttrBase<PhysicalProcessorAttr, PhysicalProcessor> {
public:
    using EnumAttrBase<PhysicalProcessorAttr, PhysicalProcessor>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

//
// DMAEngineAttr
//

class DMAEngineAttr final : public EnumAttrBase<DMAEngineAttr, DMAEngine> {
public:
    using EnumAttrBase<DMAEngineAttr, DMAEngine>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

//
// PhysicalMemoryAttr
//

class PhysicalMemoryAttr final
        : public EnumAttrBase<PhysicalMemoryAttr, PhysicalMemory> {
public:
    using EnumAttrBase<PhysicalMemoryAttr, PhysicalMemory>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

//
// ArchKindAttr
//

class ArchKindAttr final : public EnumAttrBase<ArchKindAttr, ArchKind> {
public:
    using EnumAttrBase<ArchKindAttr, ArchKind>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

//
// MemoryLocationAttr
//

class MemoryLocationAttr final
        : public EnumAttrBase<MemoryLocationAttr, MemoryLocation> {
public:
    using EnumAttrBase<MemoryLocationAttr, MemoryLocation>::EnumAttrBase;

public:
    static StringRef getMnemonic();

public:
    static MemoryLocationAttr fromPhysicalMemory(mlir::MLIRContext* ctx,
                                                 PhysicalMemory mem);

    static PhysicalMemory toPhysicalMemory(mlir::MemRefType memref);
    static PhysicalMemory toPhysicalMemory(MemoryLocation location);
    PhysicalMemory toPhysicalMemory() const;

public:
    static MemoryLocationAttr fromMemRef(mlir::MemRefType memref);

    static bool isCompatibleWith(MemoryLocation location,
                                 mlir::MemRefType memref);
    bool isCompatibleWith(mlir::MemRefType memref) const;
};

//
// ExecutionFlagAttr
//

class ExecutionFlagAttr final
        : public EnumAttrBase<ExecutionFlagAttr, ExecutionFlag> {
public:
    using EnumAttrBase<ExecutionFlagAttr, ExecutionFlag>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

//
// TaskTypeAttr
//

class TaskTypeAttr final : public EnumAttrBase<TaskTypeAttr, TaskType> {
public:
    using EnumAttrBase<TaskTypeAttr, TaskType>::EnumAttrBase;

public:
    static StringRef getMnemonic();
};

}  // namespace VPUIP
}  // namespace vpux
