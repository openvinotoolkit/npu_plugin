//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"

using namespace vpux;

// Serialization utils

const EnumMap<ov::element::Type_t, MVCNN::OVNodeType> VPUIP::mapElementType = {
        {ov::element::Type_t::undefined, MVCNN::OVNodeType::OVNodeType_UNDEFINED},
        {ov::element::Type_t::dynamic, MVCNN::OVNodeType::OVNodeType_DYNAMIC},
        {ov::element::Type_t::boolean, MVCNN::OVNodeType::OVNodeType_BOOLEAN},
        {ov::element::Type_t::bf16, MVCNN::OVNodeType::OVNodeType_BF16},
        {ov::element::Type_t::f16, MVCNN::OVNodeType::OVNodeType_F16},
        {ov::element::Type_t::f32, MVCNN::OVNodeType::OVNodeType_F32},
        {ov::element::Type_t::f64, MVCNN::OVNodeType::OVNodeType_F64},
        {ov::element::Type_t::i4, MVCNN::OVNodeType::OVNodeType_I4},
        {ov::element::Type_t::i8, MVCNN::OVNodeType::OVNodeType_I8},
        {ov::element::Type_t::i16, MVCNN::OVNodeType::OVNodeType_I16},
        {ov::element::Type_t::i32, MVCNN::OVNodeType::OVNodeType_I32},
        {ov::element::Type_t::i64, MVCNN::OVNodeType::OVNodeType_I64},
        {ov::element::Type_t::u1, MVCNN::OVNodeType::OVNodeType_U1},
        {ov::element::Type_t::u4, MVCNN::OVNodeType::OVNodeType_U4},
        {ov::element::Type_t::u8, MVCNN::OVNodeType::OVNodeType_U8},
        {ov::element::Type_t::u16, MVCNN::OVNodeType::OVNodeType_U16},
        {ov::element::Type_t::u32, MVCNN::OVNodeType::OVNodeType_U32},
        {ov::element::Type_t::u64, MVCNN::OVNodeType::OVNodeType_U64},
};

const EnumMap<vpux::PreProcessColorSpace, MVCNN::PreProcessColorSpace> vpux::VPUIP::mapPreProcessColorFormat = {
        {vpux::PreProcessColorSpace::BGR, MVCNN::PreProcessColorSpace::PreProcessColorSpace_BGR},
        {vpux::PreProcessColorSpace::RGB, MVCNN::PreProcessColorSpace::PreProcessColorSpace_RGB},
        {vpux::PreProcessColorSpace::NV12, MVCNN::PreProcessColorSpace::PreProcessColorSpace_NV12},
        {vpux::PreProcessColorSpace::I420, MVCNN::PreProcessColorSpace::PreProcessColorSpace_I420},
        {vpux::PreProcessColorSpace::NONE, MVCNN::PreProcessColorSpace::PreProcessColorSpace_DEFAULT},
};

const EnumMap<vpux::PreProcessResizeAlgorithm, MVCNN::PreProcessResizeAlgorithm>
        vpux::VPUIP::mapPreProcessResizeAlgorithm = {
                {vpux::PreProcessResizeAlgorithm::RESIZE_BILINEAR,
                 MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR},
                {vpux::PreProcessResizeAlgorithm::RESIZE_AREA,
                 MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA},
                {vpux::PreProcessResizeAlgorithm::NO_RESIZE,
                 MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE},
};

MVCNN::TargetDevice vpux::VPUIP::mapTargetDevice(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDevice::TargetDevice_KMB;
    case VPU::ArchKind::TBH:
        return MVCNN::TargetDevice::TargetDevice_TBH;
    case VPU::ArchKind::MTL:
        return MVCNN::TargetDevice::TargetDevice_MTL;
    case VPU::ArchKind::LNL:
        return MVCNN::TargetDevice::TargetDevice_LNL;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision vpux::VPUIP::mapTargetDeviceRevision(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

MVCNN::DType vpux::VPUIP::createDType(mlir::Type type) {
    if (type.isF64()) {
        return MVCNN::DType_FP64;
    } else if (type.isF32()) {
        return MVCNN::DType_FP32;
    } else if (type.isF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isBF16()) {
        return MVCNN::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isInteger(4)) {
        return MVCNN::DType_U4;
    } else if (type.isInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isInteger(1)) {
        return MVCNN::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

MVCNN::MemoryLocation vpux::VPUIP::createMemoryLocation(VPURT::BufferSection section) {
    switch (section) {
    case VPURT::BufferSection::NetworkInput:
        return MVCNN::MemoryLocation_ProgrammableInput;
    case VPURT::BufferSection::NetworkOutput:
        return MVCNN::MemoryLocation_ProgrammableOutput;
    case VPURT::BufferSection::ProfilingOutput:
        return MVCNN::MemoryLocation_ProfilingOutput;
    case VPURT::BufferSection::Constant:
        return MVCNN::MemoryLocation_GraphFile;
    case VPURT::BufferSection::SW_KernelText:
        return MVCNN::MemoryLocation_GFEmbeddedKernel;
    case VPURT::BufferSection::DDR:
        return MVCNN::MemoryLocation_VPU_DDR_Heap;
    case VPURT::BufferSection::CSRAM:
        return MVCNN::MemoryLocation_VPU_CSRAM;
    case VPURT::BufferSection::CMX_UPA:
        return MVCNN::MemoryLocation_VPU_CMX_UPA;
    case VPURT::BufferSection::CMX_NN:
        return MVCNN::MemoryLocation_VPU_CMX_NN;
    case VPURT::BufferSection::Register:
        return MVCNN::MemoryLocation_AbsoluteAddr;
    case VPURT::BufferSection::MAC_Accumulators:
        return MVCNN::MemoryLocation_MAC_Accumulators;
    default:
        VPUX_THROW("Unsupported BufferSection {0}", section);
    }
}

MVCNN::order3 vpux::VPUIP::createOrder3(mlir::ArrayAttr attr) {
    auto vec = parseIntArrayAttr<int64_t>(attr);
    std::reverse(vec.begin(), vec.end());

    VPUX_THROW_UNLESS(vec.size() <= 3, "Got wrong order array : {0}", vec);

    uint8_t x = 0, y = 0, z = 0;
    if (vec.size() >= 1) {
        x = checked_cast<uint8_t>(vec[0]);
    }
    if (vec.size() >= 2) {
        y = checked_cast<uint8_t>(vec[1]);
    }
    if (vec.size() >= 3) {
        z = checked_cast<uint8_t>(vec[2]);
    }

    return MVCNN::order3(x, y, z);
}
